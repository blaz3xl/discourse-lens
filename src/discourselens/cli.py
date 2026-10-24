from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import click
import typer
from typer.core import TyperGroup

from .analysis import (
    DEFAULT_CDA_FRAMEWORKS,
    resume_controlled_cda_review,
    run_batch_controlled_cda_review,
    run_batch_saved_controlled_profile,
    run_batch_textual_descriptors_analysis,
    run_controlled_cda_review,
    run_counterevidence_review,
    run_evidence_extraction_analysis,
    run_quote_attribution_candidate_analysis,
    run_saved_controlled_profile,
    run_textual_descriptors_analysis,
)
from .bundle import export_bundle, verify_bundle
from .config import ProjectConfig
from .corpus import (
    TOKEN,
    _tokenize_v2,
    collocation_analysis,
    collocation_association_evidence,
    collocations,
    compare_documents,
    compare_frequency_page,
    compare_named_corpora,
    document_dispersion,
    document_dispersion_evidence,
    frequency_page,
    frequency_statistic_evidence,
    keyness_page,
    kwic_evidence,
    kwic_page,
    lexical_bundle_statistic_evidence,
    ngrams,
)
from .documents import DEFAULT_OCR_DPI
from .documents import import_document as import_source
from .errors import DocumentError, ModelProviderError, ProjectError, ScopeViolationError
from .evidence_limits import MAX_CONTEXT_EVIDENCE_CHARS
from .execution import (
    build_reranking_gateway,
    execute_model_preset,
    execute_model_preset_stream,
    preview_model_preset_request,
)
from .frameworks import FRAMEWORK_REVIEWS
from .interpretation import review_finding_with_model
from .languages import inspect_local_backend, local_language_capabilities
from .models import (
    CredentialReference,
    GenerationImage,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
    ModelPrice,
)
from .nlp import load_spacy, load_stanza
from .nlp_artifacts import (
    artifact_frequency_statistic_evidence,
    dependency_patterns,
)
from .observability import configure_cli_logging
from .output import render_json, render_rows
from .profiles import ControlledCdaProfile
from .project import (
    MAX_CODEBOOK_PAGE_SIZE,
    MAX_RESEARCH_MEMO_PAGE_SIZE,
    MAX_SCOPE_REVISION_PAGE_SIZE,
    Project,
)
from .provider_catalog import supported_providers
from .reporting import (
    analysis_payload,
    comparison_payload,
    finding_payload,
    render_analysis_html,
    render_analysis_markdown,
    render_comparison_markdown,
    render_finding_html,
    render_finding_markdown,
    write_analysis_report,
)
from .scope_retrieval import MAX_FRAMEWORK_HITS
from .secrets import (
    NO_AUTH_REFERENCE,
    KeyringSecretStore,
    has_credential_like_endpoint_host,
    keyring_reference_for_credential,
)
from .skills import (
    BUILTIN_SKILLS,
    VanDijkGroupComparisonGroupInput,
    builtin_manifest,
    skill_documentation,
)
from .validation import validate_finding
from .vectors import (
    MAX_PRESET_QUERY_CHARS,
    build_paragraph_embedding_index,
    hybrid_search_with_preset,
    hybrid_search_with_provenance,
    preview_paragraph_embedding_request,
    semantic_search_with_preset,
)
from .versioning import software_version

_QUOTED_ABSOLUTE_PATH = re.compile(
    r"(?P<quote>['\"])(?P<path>(?:[A-Za-z]:[\\/]|/(?!/)|//(?!/)|"
    r"\\\\[^\\/\s]+[\\/][^\\/\s]+[\\/])[^'\"]*?)(?P=quote)"
)
_POSIX_UNC_PATH = re.compile(r"(?<![\w:/])//[^/\s'\"<>]+(?:/[^\s'\"<>]*)+")
_ABSOLUTE_PATH = re.compile(
    r"(?<![\w:/])(?:[A-Za-z]:[\\/]|/(?!/)|"
    r"\\\\[^\\/\s]+[\\/][^\\/\s]+[\\/])[^\s'\"<>]+"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])(api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|"
    r"authorization|password|secret|key|token|signature|sig)(\s*[:=]\s*)([^\s,;&]+)"
)
_BEARER_CREDENTIAL = re.compile(r"(?i)\b(bearer\s+)([^\s,;&]+)")
_SAFE_DATABASE_ERROR = (
    "Could not access the project database safely. Verify the project path, "
    "schema, and filesystem permissions."
)
_MAX_VAN_DIJK_GROUP_COMPARISON_INPUT_BYTES = 64 * 1024
_MAX_FREQUENCY_EVIDENCE_TERM_CHARS = 256
_MAX_LEXICAL_BUNDLE_EVIDENCE_CHARS = 4_096


def _safe_project_error_message(error: Exception) -> str:
    """Keep domain diagnostics useful without echoing local paths or credentials."""
    printable = "".join(char if char.isprintable() else " " for char in str(error))
    message = " ".join(printable.split()) or "The operation could not be completed."
    normalized = message.casefold()
    if normalized.startswith(("managed project storage ", "read-only project database ")) or (
        normalized == "project root is unavailable or unsafe"
    ):
        return _SAFE_DATABASE_ERROR
    message = _QUOTED_ABSOLUTE_PATH.sub("<path>", message)
    message = _POSIX_UNC_PATH.sub("<path>", message)
    message = _ABSOLUTE_PATH.sub("<path>", message)
    message = _BEARER_CREDENTIAL.sub(r"\1<redacted>", message)
    message = _SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", message)
    return message


class _DiscourseLensGroup(TyperGroup):
    """Render expected Core and local-storage failures as actionable CLI errors."""

    def invoke(self, ctx: Any) -> object:
        try:
            return super().invoke(ctx)
        except sqlite3.Error:
            click.echo(
                f"Error: {_SAFE_DATABASE_ERROR}",
                err=True,
            )
            raise click.exceptions.Exit(1) from None
        except OSError:
            # OS exception messages commonly include the full path. Keep host
            # paths and any sensitive path components out of terminal/CI logs.
            click.echo(
                "Error: Could not complete the filesystem operation. Verify the requested "
                "path and filesystem permissions.",
                err=True,
            )
            raise click.exceptions.Exit(1) from None
        except ScopeViolationError as exc:
            raise typer.BadParameter(_safe_project_error_message(exc)) from None
        except ProjectError as exc:
            click.echo(f"Error: {_safe_project_error_message(exc)}", err=True)
            raise click.exceptions.Exit(1) from None


def _version_option_callback(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
    del ctx, param
    if value:
        typer.echo(f"DiscourseLens {software_version()}")
        raise typer.Exit()
    return value


app = typer.Typer(no_args_is_help=True, cls=_DiscourseLensGroup)


@app.callback()
def configure_logging(
    ctx: typer.Context,
    _show_version: bool = typer.Option(
        False,
        "--version",
        callback=_version_option_callback,
        is_eager=True,
        help="Show the installed DiscourseLens version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Emit allowlisted JSONL lifecycle events to stderr.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug-level DiscourseLens diagnostics on stderr.",
    ),
) -> None:
    """Evidence-first discourse and corpus analysis for reproducible research."""
    restore = configure_cli_logging(verbose=verbose, debug=debug)
    ctx.call_on_close(restore)


def _ask_text_output(result: Mapping[str, object]) -> str:
    """Render a scoped ask without weakening its targeted-coverage caveat."""
    status = str(result.get("status", "unknown")).casefold()
    lines: list[str] = []
    question = result.get("question")
    if isinstance(question, str) and question:
        lines.extend(("Question", question, ""))

    if status == "no_candidates":
        lines.extend(
            (
                "No examples were found under the retrieval plan.",
                "This targeted retrieval does not establish that no such examples exist in the scope.",
            )
        )
    else:
        answer = result.get("answer_interpretation")
        if isinstance(answer, str) and answer.strip():
            lines.extend(("Answer", answer.strip()))
        alternative = result.get("alternative_interpretation")
        if isinstance(alternative, str) and alternative.strip():
            lines.extend(("", "Alternative interpretation", alternative.strip()))

    lines.extend(
        (
            "",
            "Coverage",
            "Targeted retrieval: selected passages are examples, not an exhaustive review of the scope.",
        )
    )
    coverage = result.get("coverage")
    if isinstance(coverage, Mapping):
        for key in ("coverage_mode", "selection_policy", "inspection_count", "candidate_count"):
            value = coverage.get(key)
            if value is not None:
                lines.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    elif isinstance(coverage, str) and coverage.strip():
        lines.append(coverage.strip())

    evidence = result.get("evidence")
    if isinstance(evidence, list) and evidence:
        lines.extend(("", "Sources"))
        for index, item in enumerate(evidence, start=1):
            if not isinstance(item, Mapping):
                continue
            citation = item.get("citation")
            excerpt = item.get("excerpt")
            role = item.get("role")
            heading = (
                f"{index}. {citation}" if isinstance(citation, str) and citation else f"{index}."
            )
            if isinstance(role, str) and role:
                heading += f" ({role})"
            lines.append(heading)
            if isinstance(excerpt, str) and excerpt:
                lines.append(f"   “{excerpt}”")

    limitations = result.get("limitations")
    if isinstance(limitations, (list, tuple)) and limitations:
        lines.extend(("", "Limitations"))
        lines.extend(f"- {item}" for item in limitations if isinstance(item, str) and item.strip())

    provenance_labels = (
        ("scope_snapshot_id", "Scope snapshot"),
        ("candidate_analysis_run_id", "Candidate run"),
        ("candidate_skill_run_id", "Candidate skill run"),
        ("candidate_retrieval_run_id", "Candidate retrieval"),
        ("candidate_finding_id", "Candidate finding"),
        ("answer_analysis_run_id", "Answer run"),
        ("answer_finding_id", "Answer finding"),
        ("model_execution_id", "Model execution"),
    )
    provenance = [
        f"{label}: {result[key]}"
        for key, label in provenance_labels
        if result.get(key) not in (None, "", [])
    ]
    if provenance:
        lines.extend(("", "Provenance", *provenance))
    return "\n".join(lines).rstrip()


@app.command("ask")
def ask_question(
    project: Path,
    question: str,
    revisions: list[str] = typer.Option(
        ...,
        "--revision",
        "-r",
        help="Immutable revision in the explicit scope; repeat for multiple revisions.",
    ),
    framework: str = typer.Option(
        ...,
        "--framework",
        help="Registered interpretive framework (critical-reviewer is not an ask framework).",
    ),
    preset: str = typer.Option(..., "--preset", help="Saved model preset used for this review."),
    limit: int = typer.Option(
        8,
        "--limit",
        min=1,
        max=12,
        help="Maximum retrieved evidence candidates (1..12).",
    ),
    retrieval: str = typer.Option(
        "lexical",
        "--retrieval",
        help="Candidate retrieval mode: lexical or hybrid.",
    ),
    embedding_preset: str | None = typer.Option(
        None,
        "--embedding-preset",
        help="Saved embedding preset required for hybrid retrieval.",
    ),
    embedding_artifacts: str | None = typer.Option(
        None,
        "--embedding-artifacts",
        help="Comma-separated indexed artifact IDs required for hybrid retrieval.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    """Ask a scoped framework question and receive an evidence-grounded review."""
    normalized_question = question.strip()
    if not normalized_question:
        raise typer.BadParameter("question cannot be blank", param_hint="QUESTION")
    normalized_preset = preset.strip()
    if not normalized_preset:
        raise typer.BadParameter("preset cannot be blank", param_hint="--preset")
    normalized_framework = framework.strip()
    if normalized_framework not in FRAMEWORK_REVIEWS or normalized_framework == "critical-reviewer":
        raise typer.BadParameter(
            "framework must be a registered interpretive framework",
            param_hint="--framework",
        )
    if output_format.casefold() not in {"text", "json"}:
        raise typer.BadParameter("format must be text or json", param_hint="--format")

    normalized_retrieval = retrieval.strip().casefold()
    if normalized_retrieval not in {"lexical", "hybrid"}:
        raise typer.BadParameter("retrieval must be lexical or hybrid", param_hint="--retrieval")
    normalized_embedding_preset = embedding_preset.strip() if embedding_preset is not None else None
    if normalized_retrieval == "lexical":
        if embedding_preset is not None or embedding_artifacts is not None:
            raise typer.BadParameter(
                "embedding options require --retrieval hybrid",
                param_hint="--embedding-preset/--embedding-artifacts",
            )
        normalized_embedding_artifacts: list[str] | None = None
    else:
        if normalized_embedding_preset is None or not normalized_embedding_preset:
            raise typer.BadParameter(
                "hybrid retrieval requires --embedding-preset",
                param_hint="--embedding-preset",
            )
        if embedding_artifacts is None:
            raise typer.BadParameter(
                "hybrid retrieval requires --embedding-artifacts",
                param_hint="--embedding-artifacts",
            )
        normalized_embedding_artifacts = [
            artifact.strip() for artifact in embedding_artifacts.split(",")
        ]
        if len(normalized_embedding_artifacts) > 100:
            raise typer.BadParameter(
                "provide at most 100 embedding artifact IDs",
                param_hint="--embedding-artifacts",
            )
        if any(not artifact for artifact in normalized_embedding_artifacts):
            raise typer.BadParameter(
                "embedding artifact IDs must be non-empty",
                param_hint="--embedding-artifacts",
            )
        if len(set(normalized_embedding_artifacts)) != len(normalized_embedding_artifacts):
            raise typer.BadParameter(
                "embedding artifact IDs must be unique",
                param_hint="--embedding-artifacts",
            )

    normalized_revisions = [revision.strip() for revision in revisions]
    if any(not revision for revision in normalized_revisions):
        raise typer.BadParameter("revision IDs cannot be blank", param_hint="--revision")
    if len(set(normalized_revisions)) != len(normalized_revisions):
        raise typer.BadParameter("revision IDs must be unique", param_hint="--revision")

    instance = Project.open(project)
    try:
        result = instance.ask_question(
            normalized_revisions,
            normalized_question,
            normalized_framework,
            normalized_preset,
            limit=limit,
            retrieval_mode=cast(Literal["lexical", "hybrid"], normalized_retrieval),
            **(
                {
                    "embedding_preset_name": normalized_embedding_preset,
                    "embedding_artifact_ids": normalized_embedding_artifacts,
                }
                if normalized_retrieval == "hybrid"
                else {}
            ),
        )
        if output_format.casefold() == "json":
            _emit(result)
        else:
            typer.echo(_ask_text_output(result))
    finally:
        instance.close()


project_app = typer.Typer(no_args_is_help=True, help="Create and inspect local research projects.")
document_app = typer.Typer(no_args_is_help=True, help="Import and inspect source documents.")
corpus_app = typer.Typer(no_args_is_help=True, help="Run deterministic corpus operations.")
search_app = typer.Typer(no_args_is_help=True, help="Search strictly inside an explicit scope.")
report_app = typer.Typer(no_args_is_help=True, help="Render evidence-bound research outputs.")
export_app = typer.Typer(
    no_args_is_help=True, help="Export portable, secret-free research artifacts."
)
analyze_app = typer.Typer(
    no_args_is_help=True, help="Run controlled, evidence-bound analysis pipelines."
)
provider_app = typer.Typer(
    no_args_is_help=True,
    help="Configure provider credential references and interactive OS-keyring secrets.",
)
model_app = typer.Typer(no_args_is_help=True, help="Manage reproducible model-selection presets.")
model_price_app = typer.Typer(
    no_args_is_help=True, help="Record and inspect user-supplied model price versions."
)
model_app.add_typer(model_price_app, name="price")
finding_app = typer.Typer(
    no_args_is_help=True, help="Inspect and validate evidence-bound findings."
)
evidence_app = typer.Typer(
    no_args_is_help=True, help="Store bounded, researcher-supplied evidence and context."
)
vector_app = typer.Typer(
    no_args_is_help=True, help="Build and search local derived vector indexes."
)
coding_app = typer.Typer(
    no_args_is_help=True, help="Create local codebooks and code canonical text spans."
)
nlp_app = typer.Typer(
    no_args_is_help=True, help="Inspect local language/NLP capabilities without downloads."
)
profile_app = typer.Typer(
    no_args_is_help=True, help="Create versioned declarative analysis profiles."
)
skills_app = typer.Typer(no_args_is_help=True, help="Inspect trusted built-in methodology skills.")
run_app = typer.Typer(no_args_is_help=True, help="Inspect immutable, reproducible analysis runs.")

app.add_typer(project_app, name="project")
app.add_typer(document_app, name="document")
app.add_typer(corpus_app, name="corpus")
app.add_typer(search_app, name="search")
app.add_typer(report_app, name="report")
app.add_typer(export_app, name="export")
app.add_typer(analyze_app, name="analyze")
app.add_typer(provider_app, name="providers")
app.add_typer(model_app, name="models")
app.add_typer(nlp_app, name="nlp")
app.add_typer(profile_app, name="profiles")
app.add_typer(profile_app, name="profile")
app.add_typer(skills_app, name="skills")
app.add_typer(finding_app, name="finding")
app.add_typer(evidence_app, name="evidence")
app.add_typer(vector_app, name="vectors")
app.add_typer(coding_app, name="coding")
app.add_typer(run_app, name="run")


def _emit(payload: object) -> None:
    typer.echo(render_json(payload))


def _read_context_file(path: str) -> str:
    """Read bounded UTF-8 context from a file or stdin without inline shell secrets."""
    try:
        if path == "-":
            content = sys.stdin.read(MAX_CONTEXT_EVIDENCE_CHARS + 1)
        else:
            with Path(path).open("r", encoding="utf-8") as source_file:
                content = source_file.read(MAX_CONTEXT_EVIDENCE_CHARS + 1)
    except UnicodeError as exc:
        raise ProjectError("context input must be valid UTF-8") from exc
    except OSError as exc:
        raise ProjectError("could not read context input") from exc
    if len(content) > MAX_CONTEXT_EVIDENCE_CHARS:
        raise ProjectError(
            f"context input exceeds the {MAX_CONTEXT_EVIDENCE_CHARS}-character limit"
        )
    if not content.strip():
        raise ProjectError("context input cannot be blank")
    return content


def _read_utf8_input(path: Path, *, label: str, max_characters: int | None = None) -> str:
    """Read CLI UTF-8 text, optionally bounding both bytes read and characters."""
    try:
        if max_characters is None:
            return path.read_text(encoding="utf-8")
        # UTF-8 needs at most four bytes per Unicode code point. Read one byte
        # beyond the safe maximum so a large file is rejected without loading it.
        max_bytes = max_characters * 4
        with path.open("rb") as source_file:
            raw = source_file.read(max_bytes + 1)
    except UnicodeError as exc:
        raise typer.BadParameter(f"{label} must be valid UTF-8") from exc
    except OSError as exc:
        raise typer.BadParameter(f"could not read {label}") from exc
    if len(raw) > max_bytes:
        raise typer.BadParameter(f"{label} exceeds the {max_bytes}-byte input limit")
    try:
        content = raw.decode("utf-8")
    except UnicodeError as exc:
        raise typer.BadParameter(f"{label} must be valid UTF-8") from exc
    if len(content) > max_characters:
        raise typer.BadParameter(f"{label} exceeds the {max_characters}-character limit")
    return content


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous JSON objects instead of silently taking the last key."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    """Keep the accepted input within standard JSON rather than Python extensions."""
    raise ValueError("non-standard JSON constant")


def _read_van_dijk_group_comparison_input(path: Path) -> dict[str, object]:
    """Read the fixed, bounded input contract for the deterministic comparison skill."""
    try:
        with path.open("rb") as input_file:
            raw = input_file.read(_MAX_VAN_DIJK_GROUP_COMPARISON_INPUT_BYTES + 1)
    except OSError:
        raise typer.BadParameter(
            "could not read comparison input JSON", param_hint="--input"
        ) from None
    if len(raw) > _MAX_VAN_DIJK_GROUP_COMPARISON_INPUT_BYTES:
        raise typer.BadParameter(
            "comparison input JSON exceeds the 65536-byte limit", param_hint="--input"
        )
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise typer.BadParameter(
            "comparison input JSON must be valid UTF-8", param_hint="--input"
        ) from None
    try:
        payload = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError):
        raise typer.BadParameter(
            "comparison input must contain valid JSON without duplicate keys",
            param_hint="--input",
        ) from None
    if not isinstance(payload, dict) or set(payload) != {
        "groups",
        "shared_topic_event",
        "comparability_basis",
        "codebook_id",
    }:
        raise typer.BadParameter(
            "comparison input must contain exactly groups, shared_topic_event, "
            "comparability_basis, and codebook_id",
            param_hint="--input",
        )
    for field in ("shared_topic_event", "comparability_basis", "codebook_id"):
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            raise typer.BadParameter(
                f"comparison input field {field} must be a nonempty string",
                param_hint="--input",
            )
    groups = payload["groups"]
    if not isinstance(groups, list) or len(groups) != 2:
        raise typer.BadParameter(
            "comparison input groups must contain exactly two groups", param_hint="--input"
        )
    for index, group in enumerate(groups, start=1):
        if not isinstance(group, dict) or set(group) != {"label", "annotation_ids"}:
            raise typer.BadParameter(
                f"comparison input group {index} must contain exactly label and annotation_ids",
                param_hint="--input",
            )
        label = group["label"]
        annotation_ids = group["annotation_ids"]
        if not isinstance(label, str) or not label.strip():
            raise typer.BadParameter(
                f"comparison input group {index} label must be a nonempty string",
                param_hint="--input",
            )
        if not isinstance(annotation_ids, list) or any(
            not isinstance(annotation_id, str) or not annotation_id.strip()
            for annotation_id in annotation_ids
        ):
            raise typer.BadParameter(
                f"comparison input group {index} annotation_ids must be a list of nonempty strings",
                param_hint="--input",
            )
        if len(annotation_ids) != len(set(annotation_ids)):
            raise typer.BadParameter(
                f"comparison input group {index} contains duplicate annotation IDs",
                param_hint="--input",
            )
    return cast(dict[str, object], payload)


def _validate_reranking_cli_selection(
    provider: str | None,
    credential: str | None,
    model: str | None,
    top_n: int | None,
) -> None:
    values = (provider, credential, model)
    if any(value is not None for value in values) != all(value is not None for value in values) or (
        top_n is not None and provider is None
    ):
        raise typer.BadParameter(
            "reranking requires provider, credential alias, and model together"
        )


def _validate_credential_endpoint(endpoint: str | None) -> None:
    """Reject endpoint URL forms that can carry credentials as visible CLI input."""
    if endpoint is None:
        return
    try:
        parsed = urlsplit(endpoint)
        # Accessing .port validates malformed and out-of-range ports.
        _ = parsed.port
    except ValueError:
        raise typer.BadParameter(
            "endpoint must be an absolute HTTP(S) URL without credentials, query, or fragment"
        ) from None
    if (
        endpoint != endpoint.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in endpoint)
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or has_credential_like_endpoint_host(endpoint)
    ):
        raise typer.BadParameter(
            "endpoint must be an absolute HTTP(S) URL without credentials, query, or fragment"
        )


def _model_price_source_contains_secret(value: str) -> bool:
    """Detect credential-like material in a researcher-supplied price reference."""
    if _SECRET_ASSIGNMENT.search(value) or _BEARER_CREDENTIAL.search(value):
        return True
    if re.search(
        r"(?i)[?&]\s*(?:key|token|secret|password|authorization|signature)\s*=\s*[^&#]+",
        value,
    ):
        return True
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.username is not None or parsed.password is not None


def _safe_price_source_for_cli(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectError("model price source reference is invalid")
    return "<redacted>" if _model_price_source_contains_secret(value) else value


def _spreadsheet_safe_csv_option() -> Any:
    return typer.Option(
        False,
        "--spreadsheet-safe-csv",
        help="Prefix formula-leading CSV text for spreadsheet use; requires --format csv.",
    )


def _validate_spreadsheet_safe_csv(output_format: str, enabled: bool) -> None:
    if enabled and output_format != "csv":
        raise typer.BadParameter(
            "--spreadsheet-safe-csv requires --format csv",
            param_hint="--spreadsheet-safe-csv",
        )


def _emit_rows(
    rows: Iterable[Mapping[str, object]],
    output_format: str,
    *,
    spreadsheet_safe_csv: bool = False,
    empty_csv_columns: tuple[str, ...] | None = None,
) -> None:
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if output_format == "table":
        rows = (
            {
                _escape_terminal_controls(column): _escape_terminal_controls(str(value))
                for column, value in row.items()
            }
            for row in rows
        )
    if output_format == "csv" and empty_csv_columns is not None:
        materialized_rows = list(rows)
        if not materialized_rows:
            # Keep a valid, self-describing CSV document for an empty page.
            # Column names here are fixed CLI schema identifiers, not user input.
            typer.echo(",".join(empty_csv_columns))
            return
        rows = materialized_rows
    typer.echo(
        render_rows(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv), nl=False
    )


def _escape_terminal_controls(value: str, *, preserve_newlines: bool = False) -> str:
    """Keep terminal control characters visible but inert in human-readable output."""
    escaped: list[str] = []
    for character in value:
        if character == "\n" and preserve_newlines or character.isprintable():
            escaped.append(character)
        else:
            codepoint = ord(character)
            if codepoint <= 0xFF:
                escaped.append(f"\\x{codepoint:02x}")
            elif codepoint <= 0xFFFF:
                escaped.append(f"\\u{codepoint:04x}")
            else:
                escaped.append(f"\\U{codepoint:08x}")
    return "".join(escaped)


def _echo_report_document(content: str) -> None:
    """Sanitize human-readable reports only when writing to an interactive terminal."""
    if sys.stdout.isatty():
        content = _escape_terminal_controls(content, preserve_newlines=True)
    typer.echo(content, nl=False)


def _emit_records(
    rows: Iterable[Mapping[str, object]],
    output_format: str,
    *,
    spreadsheet_safe_csv: bool = False,
) -> None:
    if output_format == "json":
        _emit([dict(row) for row in rows])
    else:
        _emit_rows(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)


def _emit_counts(
    counts,
    output_format: str,
    *,
    label: str = "token",
    spreadsheet_safe_csv: bool = False,
) -> None:
    """Keep legacy JSON pairs while offering typed rows to shell/CI formats."""
    ranked = counts.most_common()
    if output_format == "json":
        _emit(ranked)
        return
    _emit_rows(
        [{label: value, "count": count} for value, count in ranked],
        output_format,
        spreadsheet_safe_csv=spreadsheet_safe_csv,
    )


def _emit_counter_page(
    page: dict[str, object], output_format: str, *, spreadsheet_safe_csv: bool = False
) -> None:
    if output_format == "json":
        _emit(page)
        return
    rows = page["rows"]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise typer.BadParameter("frequency page rows are invalid")
    _emit_rows(
        cast(list[Mapping[str, object]], rows),
        output_format,
        spreadsheet_safe_csv=spreadsheet_safe_csv,
    )
    offset_value = page["offset"]
    total_value = page["total_values"]
    if not isinstance(offset_value, int) or not isinstance(total_value, int):
        raise typer.BadParameter("frequency page metadata is invalid")
    offset = offset_value
    total = total_value
    if rows:
        status = f"frequency rows {offset + 1}–{offset + len(rows)} of {total}"
    else:
        status = f"no frequency rows at offset {offset}; total_values={total}"
    typer.echo(f"{status}; has_more={str(page['has_more']).lower()}", err=True)


def _emit_counter_result_page(
    page: dict[str, object],
    *,
    label: str,
    output_format: str,
    spreadsheet_safe_csv: bool,
) -> None:
    if output_format == "json":
        _emit(page)
        return
    rows = cast(list[dict[str, object]], page["rows"])
    _emit_rows(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
    offset = cast(int, page["offset"])
    total = cast(int, page["total"])
    if rows:
        status = f"{label} rows {offset + 1}–{offset + len(rows)} of {total}"
    else:
        status = f"no {label} rows at offset {offset}; total={total}"
    typer.echo(f"{status}; has_more={str(page['has_more']).lower()}", err=True)


def _emit_comparison_page_rows(
    page: Mapping[str, object],
    *,
    label: str,
    output_format: str,
    spreadsheet_safe_csv: bool,
    empty_csv_columns: tuple[str, ...] | None = None,
    status_context: str | None = None,
) -> None:
    """Emit flattened comparison rows and keep pagination status off stdout."""
    rows = page.get("rows")
    offset = page.get("offset")
    total_rows = page.get("total_rows")
    has_more = page.get("has_more")
    if (
        not isinstance(rows, list)
        or not isinstance(offset, int)
        or not isinstance(total_rows, int)
        or not isinstance(has_more, bool)
    ):
        raise typer.BadParameter("comparison page metadata is invalid")
    if any(not isinstance(row, Mapping) for row in rows):
        raise typer.BadParameter("comparison page rows are invalid")
    _emit_rows(
        cast(list[Mapping[str, object]], rows),
        output_format,
        spreadsheet_safe_csv=spreadsheet_safe_csv,
        empty_csv_columns=empty_csv_columns,
    )
    if rows:
        status = f"{label} rows {offset + 1}–{offset + len(rows)} of {total_rows}"
    else:
        status = f"no {label} rows at offset {offset}; total_rows={total_rows}"
    if status_context:
        status = f"{status}; {status_context}"
    typer.echo(f"{status}; has_more={str(has_more).lower()}", err=True)


def _init_project(path: Path) -> None:
    Project.init(path).close()
    _emit({"project": str(path), "status": "created"})


@app.command()
def init(path: Path = typer.Argument(...)):
    """Legacy alias for `project init`."""
    _init_project(path)


@project_app.command("init")
def project_init(path: Path = typer.Argument(...)) -> None:
    """Create a portable local DiscourseLens project."""
    _init_project(path)


@project_app.command("offline")
def project_offline(
    project: Path,
    enabled: bool = typer.Option(..., help="Enable or disable remote-operation blocking."),
) -> None:
    """Set persistent offline mode; deterministic local corpus tools remain available."""
    instance = Project.open(project)
    try:
        instance.set_offline_mode(enabled)
        _emit({"project_id": instance.project_id, "offline_mode": instance.offline_mode})
    finally:
        instance.close()


@project_app.command("config")
def project_config(project: Path) -> None:
    """Show effective validated import configuration without secret values."""
    instance = Project.open(project)
    try:
        _emit(instance.config.model_dump())
    finally:
        instance.close()


@project_app.command("set-import-limit")
def project_set_import_limit(project: Path, bytes: int = typer.Argument(..., min=1)) -> None:
    """Set a safe source-file size maximum for later local document imports."""
    instance = Project.open(project)
    try:
        instance.set_project_config(ProjectConfig(import_max_bytes=bytes))
        _emit(instance.config.model_dump())
    finally:
        instance.close()


@run_app.command("list")
def run_list(
    project: Path,
    limit: int = typer.Option(100, min=1, max=1000),
    offset: int | None = typer.Option(
        None,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for a paged run listing.",
    ),
) -> None:
    """List immutable AnalysisRuns without exposing prompts, responses or secrets."""
    instance = Project.open(project)
    try:
        if offset is None:
            _emit(instance.list_analysis_runs(limit))
        else:
            _emit(instance.analysis_runs_page(offset=offset, limit=limit))
    finally:
        instance.close()


@run_app.command("inspect")
def run_inspect(
    project: Path,
    analysis_run: str,
    offset: int | None = typer.Option(
        None,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for paged run provenance.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        max=100,
        help="Maximum rows per provenance collection (1..100).",
    ),
) -> None:
    """Inspect historical run configuration, scope, skill and model execution provenance."""
    instance = Project.open(project)
    try:
        if offset is None and limit is None:
            _emit(instance.analysis_run_summary(analysis_run))
        else:
            _emit(
                instance.analysis_run_summary_page(
                    analysis_run,
                    offset=0 if offset is None else offset,
                    limit=100 if limit is None else limit,
                )
            )
    finally:
        instance.close()


@app.command("import")
def import_document(
    project: Path,
    file: Path,
    ocr_language: str | None = typer.Option(
        None,
        "--ocr-language",
        help="Explicitly run local Tesseract OCR for PDF text/image areas (for example eng+spa).",
    ),
    ocr_dpi: int = typer.Option(DEFAULT_OCR_DPI, min=72, max=600),
    max_bytes: int | None = typer.Option(
        None,
        "--max-bytes",
        min=1,
        help="Override the configured source-file size limit for this import.",
    ),
):
    """Legacy alias for `document import`."""
    instance = Project.open(project)
    try:
        _emit(
            import_source(
                instance,
                file,
                max_bytes=max_bytes,
                ocr_language=ocr_language,
                ocr_dpi=ocr_dpi,
            ).__dict__
        )
    finally:
        instance.close()


@document_app.command("import")
def document_import(
    project: Path,
    file: Path,
    ocr_language: str | None = typer.Option(
        None,
        "--ocr-language",
        help="Explicitly run local Tesseract OCR for PDF text/image areas (for example eng+spa).",
    ),
    ocr_dpi: int = typer.Option(DEFAULT_OCR_DPI, min=72, max=600),
    max_bytes: int | None = typer.Option(
        None,
        "--max-bytes",
        min=1,
        help="Override the configured source-file size limit for this import.",
    ),
) -> None:
    """Import UTF-8 text/Markdown/HTML, DOCX, or PDF; OCR is opt-in and local."""
    import_document(project, file, ocr_language, ocr_dpi, max_bytes)


@document_app.command("import-many")
def document_import_many(
    project: Path,
    files: list[Path] = typer.Argument(..., help="Files to import, processed in this order."),
    ocr_language: str | None = typer.Option(
        None,
        "--ocr-language",
        help="Explicitly run local Tesseract OCR for PDF text/image areas (for example eng+spa).",
    ),
    ocr_dpi: int = typer.Option(DEFAULT_OCR_DPI, min=72, max=600),
    max_bytes: int | None = typer.Option(
        None,
        "--max-bytes",
        min=1,
        help="Override the configured source-file size limit for each file in this batch.",
    ),
) -> None:
    """Import several files in order, returning an outcome for each input."""
    instance = Project.open(project)
    try:
        result = instance.import_documents(
            files,
            max_bytes=max_bytes,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
        )
        _emit(
            {
                "total_count": result.total_count,
                "success_count": result.success_count,
                "duplicate_count": result.duplicate_count,
                "failed_count": result.failed_count,
                "items": [
                    {
                        "index": item.index,
                        "source_name": files[item.index].name,
                        "status": item.status,
                        "document_id": item.document_id,
                        "revision_id": item.revision_id,
                        "failure_code": item.failure_code,
                        "warnings": list(item.warnings),
                    }
                    for item in result.items
                ],
            }
        )
        if result.failed_count:
            raise typer.Exit(code=1)
    finally:
        instance.close()


@document_app.command("reprocess")
def document_reprocess(
    project: Path,
    document_id: str,
    file: Path,
    extractor_suffix: str = typer.Option(
        ".reprocessed",
        "--extractor-suffix",
        help="Suffix added to the PDF extractor version recorded for this revision.",
    ),
    ocr_language: str | None = typer.Option(
        None,
        "--ocr-language",
        help="Explicitly run local Tesseract OCR for PDF text/image areas (for example eng+spa).",
    ),
    ocr_dpi: int = typer.Option(DEFAULT_OCR_DPI, min=72, max=600),
    max_bytes: int | None = typer.Option(
        None,
        "--max-bytes",
        min=1,
        help="Override the configured source-file size limit for this reprocessing operation.",
    ),
) -> None:
    """Create a new PDF revision for an existing document using local extraction."""
    instance = Project.open(project)
    try:
        result = instance.reprocess_document(
            document_id,
            file,
            extractor_suffix=extractor_suffix,
            max_bytes=max_bytes,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
        )
        _emit(asdict(result))
        if result.status not in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
            if result.status == "OCR_REQUIRED":
                message = "PDF extraction found no text; use --ocr-language to enable local OCR."
            elif result.status == "PASSWORD_REQUIRED":
                message = "The PDF is password-protected and cannot be reprocessed."
            else:
                message = f"PDF reprocessing did not complete (status: {result.status})."
            typer.echo(f"Error: {message}", err=True)
            raise typer.Exit(code=1)
    finally:
        instance.close()


@document_app.command("list")
def document_list(
    project: Path,
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for the document page.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum documents in this page (1..1000).",
    ),
) -> None:
    """List bounded document metadata and latest revisions without source text."""
    instance = Project.open(project)
    try:
        _emit(instance.documents_page(offset=offset, limit=limit))
    finally:
        instance.close()


@document_app.command("show")
def document_show(project: Path, revision: str) -> None:
    """Show immutable document-revision provenance without exposing absolute paths."""
    instance = Project.open(project)
    try:
        try:
            summary = instance.document_revision_summary(revision)
        except DocumentError as exc:
            if str(exc) == "unknown revision":
                raise typer.BadParameter("unknown revision", param_hint="--revision") from None
            raise
        _emit(summary)
    finally:
        instance.close()


@document_app.command("page")
def document_page(
    project: Path,
    revision_id: str,
    page_index: int = typer.Argument(..., min=0),
    max_chars: int = typer.Option(
        8_000,
        "--max-chars",
        min=1,
        max=10_000,
        help="Maximum page text characters to return (1..10000).",
    ),
) -> None:
    """Read one persisted page by immutable revision and zero-based index."""
    instance = Project.open(project)
    try:
        page = dict(instance.page(revision_id, page_index))
        text = page["text"]
        page["text"] = text[:max_chars]
        page["text_truncated"] = len(text) > max_chars
        _emit(page)
    finally:
        instance.close()


@document_app.command("paragraph")
def document_paragraph(
    project: Path,
    paragraph_id: str,
    max_chars: int = typer.Option(
        8_000,
        "--max-chars",
        min=1,
        max=10_000,
        help="Maximum paragraph text characters to return (1..10000).",
    ),
) -> None:
    """Read one persisted paragraph by canonical ID."""
    instance = Project.open(project)
    try:
        paragraph = dict(instance.paragraph(paragraph_id))
        text = paragraph["text"]
        paragraph["text"] = text[:max_chars]
        paragraph["text_truncated"] = len(text) > max_chars
        _emit(paragraph)
    finally:
        instance.close()


@document_app.command("context")
def document_context(
    project: Path,
    span_id: str,
    before: int = typer.Option(
        200,
        "--before",
        min=0,
        max=10_000,
        help="Characters of preceding context (0..10000).",
    ),
    after: int = typer.Option(
        200,
        "--after",
        min=0,
        max=10_000,
        help="Characters of following context (0..10000).",
    ),
    max_chars: int = typer.Option(
        8_000,
        "--max-chars",
        min=1,
        max=10_000,
        help="Maximum exact-span text characters to return (1..10000).",
    ),
) -> None:
    """Read bounded Core-resolved context around a canonical source span."""
    instance = Project.open(project)
    try:
        context = instance.span_context(span_id, before=before, after=after)
        exact_text = context["exact_text"]
        if not isinstance(exact_text, str):
            raise ProjectError("source context returned invalid exact text")
        context["exact_text"] = exact_text[:max_chars]
        context["exact_text_truncated"] = len(exact_text) > max_chars
        _emit(context)
    finally:
        instance.close()


@document_app.command("set-metadata")
def document_set_metadata(project: Path, document: str, key: str, value_json: str) -> None:
    """Set JSON metadata on a document without mutating its immutable revisions."""
    try:
        value = json.loads(value_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("value_json must be valid JSON", param_hint="value_json") from exc
    instance = Project.open(project)
    try:
        instance.set_document_metadata(document, key, value)
        _emit({"document_id": document, "key": key, "status": "updated"})
    finally:
        instance.close()


@document_app.command("metadata")
def document_metadata(project: Path, document: str) -> None:
    """Show structured metadata for one document."""
    instance = Project.open(project)
    try:
        _emit(instance.document_metadata(document))
    finally:
        instance.close()


@document_app.command("select-metadata")
def document_select_metadata(project: Path, filters_json: str = typer.Option(...)) -> None:
    """Freeze latest revisions matching every exact document-metadata filter."""
    try:
        filters = json.loads(filters_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("filters_json must be a JSON object") from exc
    if not isinstance(filters, dict):
        raise typer.BadParameter("filters_json must be a JSON object")
    instance = Project.open(project)
    try:
        scope_id, revisions = instance.snapshot_by_document_metadata(filters)
        _emit({"scope_id": scope_id, "revision_ids": revisions, "filters": filters})
    finally:
        instance.close()


def frequency_cmd(
    project: Path,
    revision: str,
    output_format: str = "json",
    *,
    offset: int = 0,
    limit: int = 100,
    spreadsheet_safe_csv: bool = False,
):
    instance = Project.open(project)
    try:
        result = frequency_page(
            instance,
            instance.snapshot([revision]),
            offset=offset,
            limit=limit,
        )
        if output_format == "json":
            _emit(result)
        else:
            _emit_rows(
                result["rows"],
                output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
                empty_csv_columns=("token", "count"),
            )
            if result["rows"]:
                status = (
                    f"frequency rows {offset + 1}–{offset + len(result['rows'])} "
                    f"of {result['total_types']}"
                )
            else:
                status = (
                    f"no frequency rows at offset {offset}; total_types={result['total_types']}"
                )
            typer.echo(
                f"{status}; total_tokens={result['total_tokens']}; "
                f"has_more={str(result['has_more']).lower()}",
                err=True,
            )
    finally:
        instance.close()


@corpus_app.command("frequency")
def corpus_frequency(
    project: Path,
    revision: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, help="Zero-based token-type offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum rows in this page."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return a bounded surface-token frequency page for one document revision."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    frequency_cmd(
        project,
        revision,
        output_format,
        offset=offset,
        limit=limit,
        spreadsheet_safe_csv=spreadsheet_safe_csv,
    )


@corpus_app.command("frequency-scope")
def corpus_frequency_scope(
    project: Path,
    scope_id: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based token-type offset.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum rows in this page (1..1000).",
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return a bounded surface-token frequency page for an existing scope."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")

    instance = Project.open(project, read_only=True)
    try:
        try:
            page = instance.corpus_frequency_page(scope_id, offset=offset, limit=limit)
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise
        if output_format == "json":
            _emit(page)
            return

        rows = cast(list[Mapping[str, object]], page["rows"])
        _emit_rows(
            rows,
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
            empty_csv_columns=("token", "count"),
        )
        if rows:
            status = f"frequency rows {offset + 1}–{offset + len(rows)} of {page['total_types']}"
        else:
            status = f"no frequency rows at offset {offset}; total_types={page['total_types']}"
        typer.echo(
            f"{status}; scope_id={scope_id}; limit={limit}; "
            f"total_tokens={page['total_tokens']}; has_more={str(page['has_more']).lower()}",
            err=True,
        )
    finally:
        instance.close()


def kwic_cmd(
    project: Path,
    revision: str,
    query: str,
    output_format: str = "json",
    *,
    offset: int = 0,
    limit: int = 100,
    spreadsheet_safe_csv: bool = False,
):
    instance = Project.open(project)
    try:
        result = kwic_page(
            instance,
            instance.snapshot([revision]),
            query,
            offset=offset,
            limit=limit,
        )
        if output_format == "json":
            _emit(result)
        else:
            _emit_rows(
                [dict(row) for row in result["rows"]],
                output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
            )
            if result["rows"]:
                status = (
                    f"KWIC rows {offset + 1}–{offset + len(result['rows'])} of "
                    f"{result['total_count']}"
                )
            else:
                status = f"No KWIC rows at offset {offset}; total={result['total_count']}"
            typer.echo(f"{status}; has_more={str(result['has_more']).lower()}", err=True)
    finally:
        instance.close()


@corpus_app.command("frequency-evidence")
def corpus_frequency_evidence(project: Path, revision: str, term: str) -> None:
    """Persist an exhaustive deterministic token frequency as reusable evidence."""
    instance = Project.open(project)
    try:
        _emit(frequency_statistic_evidence(instance, instance.snapshot([revision]), term))
    finally:
        instance.close()


@corpus_app.command("frequency-evidence-scope")
def corpus_frequency_evidence_scope(project: Path, scope_id: str, term: str) -> None:
    """Persist exhaustive surface-frequency evidence for an existing scope."""
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    if not term.strip():
        raise typer.BadParameter("term cannot be blank", param_hint="TERM")
    if len(term) > _MAX_FREQUENCY_EVIDENCE_TERM_CHARS:
        raise typer.BadParameter(
            f"term must be at most {_MAX_FREQUENCY_EVIDENCE_TERM_CHARS} characters",
            param_hint="TERM",
        )
    if len(list(_tokenize_v2(term))) != 1:
        raise typer.BadParameter("term must contain exactly one surface token", param_hint="TERM")

    instance = Project.open(project)
    try:
        try:
            result = instance.corpus_frequency_evidence(scope_id, term)
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise
        _emit(result)
    finally:
        instance.close()


@corpus_app.command("dispersion")
def corpus_dispersion(
    project: Path,
    revisions: str,
    term: str,
    offset: int | None = typer.Option(None, min=0, help="Zero-based document row offset."),
    limit: int | None = typer.Option(
        None, min=1, max=1_000, help="Maximum document rows to return (1..1000)."
    ),
) -> None:
    """Show exhaustive dispersion totals, optionally with one document page."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        scope_id = instance.snapshot(revision_ids)
        if offset is None and limit is None:
            result = document_dispersion(instance, scope_id, term)
        else:
            result = instance.corpus_dispersion_page(
                scope_id,
                term,
                offset=0 if offset is None else offset,
                limit=100 if limit is None else limit,
            )
        _emit(result)
    finally:
        instance.close()


@corpus_app.command("dispersion-scope")
def corpus_dispersion_scope(
    project: Path,
    scope_id: str,
    term: str,
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based document row offset.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum document rows to return (1..1000).",
    ),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Show exhaustive dispersion totals and a page from an existing scope."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    if not term.strip():
        raise typer.BadParameter("term cannot be blank", param_hint="TERM")
    if not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF:
        raise typer.BadParameter(
            "offset must be in the non-negative signed 64-bit range", param_hint="--offset"
        )

    instance = Project.open(project, read_only=True)
    try:
        try:
            page = instance.corpus_dispersion_page(
                scope_id, term, offset=offset, limit=limit
            )
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise

        if output_format == "json":
            _emit(page)
            return

        documents = cast(list[Mapping[str, object]], page["documents"])
        _emit_rows(
            documents,
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
            empty_csv_columns=("revision_id", "occurrences", "token_count", "per_1000_tokens"),
        )
        if documents:
            status = f"dispersion document rows {offset + 1}–{offset + len(documents)} of {page['total_rows']}"
        else:
            status = f"no dispersion document rows at offset {offset}; total_rows={page['total_rows']}"
        typer.echo(
            f"{status}; scope_id={scope_id}; term={page['term']}; limit={limit}; "
            f"document_frequency={page['document_frequency']}; "
            f"total_occurrences={page['total_occurrences']}; "
            f"has_more={str(page['has_more']).lower()}",
            err=True,
        )
    finally:
        instance.close()


@corpus_app.command("dispersion-evidence")
def corpus_dispersion_evidence(project: Path, revisions: str, term: str) -> None:
    """Persist the exhaustive document distribution as citable corpus evidence."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        _emit(document_dispersion_evidence(instance, instance.snapshot(revision_ids), term))
    finally:
        instance.close()


@corpus_app.command("artifact-frequency-evidence")
def corpus_artifact_frequency_evidence(
    project: Path,
    revision: str,
    artifact: list[str] = typer.Option(
        ..., "--artifact", help="One artifact ID for this revision."
    ),
    field: str = typer.Option(
        ..., "--field", help="normalized, lemma, pos, or dependency_relation."
    ),
    value: str = typer.Option(..., "--value", help="Exact derived annotation value to count."),
) -> None:
    """Persist exhaustive evidence from complete, versioned local NLP artifacts."""
    instance = Project.open(project)
    try:
        _emit(
            artifact_frequency_statistic_evidence(
                instance, instance.snapshot([revision]), artifact, field=field, value=value
            )
        )
    finally:
        instance.close()


@corpus_app.command("kwic")
def corpus_kwic(
    project: Path,
    revision: str,
    query: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, help="Zero-based result offset."),
    limit: int = typer.Option(100, min=1, max=1000, help="Maximum rows to return (1..1000)."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return one bounded KWIC page in exactly one explicit document revision."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    kwic_cmd(
        project,
        revision,
        query,
        output_format,
        offset=offset,
        limit=limit,
        spreadsheet_safe_csv=spreadsheet_safe_csv,
    )


@corpus_app.command("kwic-scope")
def corpus_kwic_scope(
    project: Path,
    scope_id: str,
    query: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    window: int = typer.Option(
        5,
        "--window",
        min=0,
        max=100,
        help="Tokens of left and right context (0..100).",
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based result offset.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum rows in this page (1..1000).",
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return a bounded KWIC page for a query inside an existing scope."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")

    instance = Project.open(project, read_only=True)
    try:
        try:
            page = instance.corpus_kwic_page(
                scope_id, query, window=window, offset=offset, limit=limit
            )
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise
        if output_format == "json":
            _emit(page)
            return

        rows = cast(list[Mapping[str, object]], page["rows"])
        _emit_rows(
            rows,
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
            empty_csv_columns=(
                "revision_id",
                "start_offset",
                "end_offset",
                "left",
                "keyword",
                "right",
            ),
        )
        if rows:
            status = f"KWIC rows {offset + 1}–{offset + len(rows)} of {page['total_count']}"
        else:
            status = f"No KWIC rows at offset {offset}; total={page['total_count']}"
        typer.echo(
            f"{status}; scope_id={scope_id}; window={window}; limit={limit}; "
            f"has_more={str(page['has_more']).lower()}",
            err=True,
        )
    finally:
        instance.close()


@corpus_app.command("kwic-evidence")
def corpus_kwic_evidence(
    project: Path,
    revision: str,
    query: str,
    start_offset: int,
    end_offset: int,
) -> None:
    """Persist evidence for a KWIC hit using its returned canonical offsets."""
    instance = Project.open(project)
    try:
        scope_id = instance.snapshot([revision])
        _emit(
            kwic_evidence(
                instance,
                scope_id,
                revision,
                query,
                start_offset,
                end_offset,
            )
        )
    finally:
        instance.close()


@corpus_app.command("kwic-evidence-scope")
def corpus_kwic_evidence_scope(
    project: Path,
    scope_id: str,
    revision_id: str,
    query: str,
    start_offset: int,
    end_offset: int,
) -> None:
    """Persist evidence for a KWIC hit in an existing immutable scope."""
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    if not revision_id.strip():
        raise typer.BadParameter("revision ID cannot be blank", param_hint="REVISION_ID")
    if not query.strip():
        raise typer.BadParameter("query cannot be blank", param_hint="QUERY")
    if start_offset < 0 or end_offset <= start_offset:
        raise typer.BadParameter(
            "offsets require START_OFFSET >= 0 and END_OFFSET > START_OFFSET",
            param_hint="START_OFFSET/END_OFFSET",
        )

    instance = Project.open(project)
    try:
        _emit(
            instance.corpus_kwic_evidence(
                scope_id,
                revision_id,
                query,
                start_offset,
                end_offset,
            )
        )
    finally:
        instance.close()


@corpus_app.command("ngrams")
def corpus_ngrams(
    project: Path,
    revision: str,
    size: int = typer.Option(..., "--size", min=1, help="Number of surface tokens per n-gram."),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int | None = typer.Option(
        None, "--offset", min=0, max=1_000_000, help="Zero-based page offset."
    ),
    limit: int | None = typer.Option(
        None, "--limit", min=1, max=1_000, help="Maximum rows in this page."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Count deterministic surface n-grams inside one explicit document revision."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        scope_id = instance.snapshot([revision])
        if offset is not None or limit is not None:
            core_page = instance.corpus_ngram_page(
                scope_id,
                size,
                offset=0 if offset is None else offset,
                limit=100 if limit is None else limit,
            )
            page = {
                "offset": core_page["offset"],
                "limit": core_page["limit"],
                "total": core_page["total_values"],
                "has_more": core_page["has_more"],
                "rows": [
                    {"ngram": row["ngram"], "count": row["count"]} for row in core_page["rows"]
                ],
            }
            _emit_counter_result_page(
                page,
                label="n-gram",
                output_format=output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
            )
            return
        result = ngrams(instance, scope_id, size)
        if output_format == "json":
            _emit([[" ".join(terms), count] for terms, count in result.most_common()])
        else:
            _emit_rows(
                [
                    {"ngram": " ".join(terms), "count": count}
                    for terms, count in result.most_common()
                ],
                output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
            )
    finally:
        instance.close()


@corpus_app.command("ngrams-scope")
def corpus_ngrams_scope(
    project: Path,
    scope_id: str,
    size: int = typer.Option(..., "--size", min=1, help="Number of surface tokens per n-gram."),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=1_000_000,
        help="Zero-based page offset.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum rows in this page (1..1000).",
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return a bounded surface n-gram page from an existing immutable scope."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")

    instance = Project.open(project, read_only=True)
    try:
        try:
            page = instance.corpus_ngram_page(scope_id, size, offset=offset, limit=limit)
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise
        if output_format == "json":
            _emit(page)
            return

        rows = cast(list[Mapping[str, object]], page["rows"])
        _emit_rows(
            rows,
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
            empty_csv_columns=("ngram", "count"),
        )
        if rows:
            status = f"n-gram rows {offset + 1}–{offset + len(rows)} of {page['total_values']}"
        else:
            status = f"no n-gram rows at offset {offset}; total_values={page['total_values']}"
        typer.echo(
            f"{status}; scope_id={scope_id}; size={size}; limit={limit}; "
            f"has_more={str(page['has_more']).lower()}",
            err=True,
        )
    finally:
        instance.close()


@corpus_app.command("lexical-bundles")
def corpus_lexical_bundles(
    project: Path,
    revisions: str,
    size: int = typer.Option(3, "--size", min=2, max=10),
    min_frequency: int = typer.Option(2, "--min-frequency", min=1),
    min_documents: int = typer.Option(1, "--min-documents", min=1),
    offset: int = typer.Option(0, "--offset", min=0, help="Zero-based result offset."),
    limit: int = typer.Option(100, "--limit", min=1, max=1_000),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List recurrent surface bundles within an explicit revision selection."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
        scope_id = instance.snapshot(revision_ids)
        result = instance.corpus_lexical_bundles(
            scope_id,
            size=size,
            min_frequency=min_frequency,
            min_document_frequency=min_documents,
            offset=offset,
            limit=limit,
        )
        if output_format == "json":
            _emit(result)
        else:
            _emit_rows(result["rows"], output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
            if result["rows"]:
                status = (
                    f"lexical bundle rows {offset + 1}–{offset + len(result['rows'])} "
                    f"of {result['candidate_count']}"
                )
            else:
                status = (
                    f"no lexical bundle rows at offset {offset}; "
                    f"candidate_count={result['candidate_count']}"
                )
            typer.echo(f"{status}; has_more={str(result['has_more']).lower()}", err=True)
    finally:
        instance.close()


@corpus_app.command("lexical-bundles-scope")
def corpus_lexical_bundles_scope(
    project: Path,
    scope_id: str,
    size: int = typer.Option(3, "--size", min=2, max=10),
    min_frequency: int = typer.Option(2, "--min-frequency", min=1),
    min_documents: int = typer.Option(1, "--min-documents", min=1),
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based result offset.",
    ),
    limit: int = typer.Option(100, "--limit", min=1, max=1_000),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List recurrent surface bundles from an existing immutable scope."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    if not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF:
        raise typer.BadParameter(
            "offset must be in the non-negative signed 64-bit range", param_hint="--offset"
        )

    instance = Project.open(project, read_only=True)
    try:
        try:
            result = instance.corpus_lexical_bundles(
                scope_id,
                size=size,
                min_frequency=min_frequency,
                min_document_frequency=min_documents,
                offset=offset,
                limit=limit,
            )
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise

        if output_format == "json":
            _emit(result)
            return

        rows = cast(list[Mapping[str, object]], result["rows"])
        _emit_rows(
            rows,
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
            empty_csv_columns=("bundle", "frequency", "document_frequency"),
        )
        if rows:
            status = (
                f"lexical bundle rows {offset + 1}–{offset + len(rows)} "
                f"of {result['candidate_count']}"
            )
        else:
            status = f"no lexical bundle rows at offset {offset}"
        typer.echo(
            f"{status}; scope_id={scope_id}; size={size}; "
            f"min_frequency={min_frequency}; min_documents={min_documents}; limit={limit}; "
            f"candidate_count={result['candidate_count']}; "
            f"has_more={str(result['has_more']).lower()}",
            err=True,
        )
    finally:
        instance.close()


@corpus_app.command("lexical-bundle-evidence")
def corpus_lexical_bundle_evidence(
    project: Path,
    revisions: str,
    bundle: str,
    size: int = typer.Option(..., "--size", min=2, max=10),
) -> None:
    """Persist an exhaustive, citable count for one exact surface bundle."""
    instance = Project.open(project)
    try:
        revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
        result = lexical_bundle_statistic_evidence(
            instance, instance.snapshot(revision_ids), bundle, size=size
        )
        _emit(result)
    finally:
        instance.close()


@corpus_app.command("lexical-bundle-evidence-scope")
def corpus_lexical_bundle_evidence_scope(
    project: Path,
    scope_id: str,
    bundle: str,
    size: int = typer.Option(..., "--size", min=2, max=10),
) -> None:
    """Persist citable exact lexical-bundle evidence for an existing scope."""
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    if not bundle.strip():
        raise typer.BadParameter("bundle cannot be blank", param_hint="BUNDLE")
    if len(bundle) > _MAX_LEXICAL_BUNDLE_EVIDENCE_CHARS:
        raise typer.BadParameter(
            f"bundle must be at most {_MAX_LEXICAL_BUNDLE_EVIDENCE_CHARS} characters",
            param_hint="BUNDLE",
        )
    if not 2 <= size <= 10:
        raise typer.BadParameter("size must be between 2 and 10", param_hint="--size")
    if len(TOKEN.findall(bundle)) != size:
        raise typer.BadParameter(
            f"bundle must contain exactly {size} surface tokens", param_hint="BUNDLE"
        )

    instance = Project.open(project)
    try:
        try:
            result = instance.corpus_lexical_bundle_evidence(scope_id, bundle, size=size)
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise
        _emit(result)
    finally:
        instance.close()


def collocations_cmd(
    project: Path,
    revision: str,
    query: str,
    window: int = 5,
    output_format: str = "json",
    *,
    spreadsheet_safe_csv: bool = False,
    offset: int | None = None,
    limit: int | None = None,
):
    instance = Project.open(project)
    try:
        scope_id = instance.snapshot([revision])
        if offset is not None or limit is not None:
            core_page = instance.corpus_collocations_page(
                scope_id,
                query,
                window=window,
                offset=0 if offset is None else offset,
                limit=100 if limit is None else limit,
            )
            page = {
                "offset": core_page["offset"],
                "limit": core_page["limit"],
                "total": core_page["total_values"],
                "has_more": core_page["has_more"],
                "rows": [
                    {"collocate": row["term"], "count": row["count"]} for row in core_page["rows"]
                ],
            }
            _emit_counter_result_page(
                page,
                label="collocation",
                output_format=output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
            )
        else:
            result = collocations(instance, scope_id, query, window)
            _emit_counts(result, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
    finally:
        instance.close()


@corpus_app.command("collocations")
def corpus_collocations(
    project: Path,
    revision: str,
    query: str,
    window: int = typer.Option(
        5,
        "--window",
        min=1,
        max=100,
        help="Token radius around each query occurrence (1..100).",
    ),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int | None = typer.Option(
        None, "--offset", min=0, max=1_000_000, help="Zero-based page offset."
    ),
    limit: int | None = typer.Option(
        None, "--limit", min=1, max=1_000, help="Maximum rows in this page."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return deterministic window collocations in an explicit revision."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    collocations_cmd(
        project,
        revision,
        query,
        window,
        output_format,
        spreadsheet_safe_csv=spreadsheet_safe_csv,
        offset=offset,
        limit=limit,
    )


@corpus_app.command("collocations-scope")
def corpus_collocations_scope(
    project: Path,
    scope_id: str,
    query: str,
    window: int = typer.Option(
        5,
        "--window",
        min=1,
        max=100,
        help="Token radius around each query occurrence (1..100).",
    ),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=1_000_000,
        help="Zero-based page offset.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum rows in this page (1..1000).",
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return a bounded collocation page from an existing immutable scope."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    if not query.strip():
        raise typer.BadParameter("query cannot be blank", param_hint="QUERY")

    instance = Project.open(project, read_only=True)
    try:
        try:
            page = instance.corpus_collocations_page(
                scope_id,
                query,
                window=window,
                offset=offset,
                limit=limit,
            )
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise
        if output_format == "json":
            _emit(page)
            return

        rows = cast(list[Mapping[str, object]], page["rows"])
        _emit_rows(
            rows,
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
            empty_csv_columns=("term", "count"),
        )
        if rows:
            status = f"collocation rows {offset + 1}–{offset + len(rows)} of {page['total_values']}"
        else:
            status = f"no collocation rows at offset {offset}; total_values={page['total_values']}"
        typer.echo(
            f"{status}; scope_id={scope_id}; window={window}; limit={limit}; "
            f"has_more={str(page['has_more']).lower()}",
            err=True,
        )
    finally:
        instance.close()


@corpus_app.command("collocation-analysis")
def corpus_collocation_analysis(
    project: Path,
    revisions: str,
    query: str,
    window: int = typer.Option(5, min=1, max=100),
    min_frequency: int = typer.Option(2, min=1),
    offset: int = typer.Option(0, min=0, max=1_000_000, help="Zero-based ranked result offset."),
    limit: int = typer.Option(100, min=1, max=10_000),
) -> None:
    """Return a ranked page of window collocates with descriptive MI and t-score measures."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        scope_id = instance.snapshot(revision_ids)
        _emit(
            collocation_analysis(
                instance,
                scope_id,
                query,
                window=window,
                min_frequency=min_frequency,
                offset=offset,
                limit=limit,
            )
        )
    finally:
        instance.close()


@corpus_app.command("collocation-evidence")
def corpus_collocation_evidence(
    project: Path,
    revisions: str,
    query: str,
    collocate: str,
    window: int = typer.Option(5, min=1, max=100),
    min_frequency: int = typer.Option(2, min=1),
) -> None:
    """Persist one exhaustive collocation association as statistic evidence."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        _emit(
            collocation_association_evidence(
                instance,
                instance.snapshot(revision_ids),
                query,
                collocate,
                window=window,
                min_frequency=min_frequency,
            )
        )
    finally:
        instance.close()


@corpus_app.command("collocation-evidence-scope")
def corpus_collocation_evidence_scope(
    project: Path,
    scope_id: str,
    query: str,
    collocate: str,
    window: int = typer.Option(5, "--window", min=1, max=100),
    min_frequency: int = typer.Option(2, "--min-frequency", min=1),
) -> None:
    """Persist a scope-bound association; MI and t-score are descriptive, not significance tests."""
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    for label, value in (("query", query), ("collocate", collocate)):
        if not value.strip():
            raise typer.BadParameter(f"{label} cannot be blank", param_hint=label.upper())
        if len(value) > _MAX_FREQUENCY_EVIDENCE_TERM_CHARS:
            raise typer.BadParameter(
                f"{label} must be at most {_MAX_FREQUENCY_EVIDENCE_TERM_CHARS} characters",
                param_hint=label.upper(),
            )
        if len(list(_tokenize_v2(value))) != 1:
            raise typer.BadParameter(
                f"{label} must contain exactly one surface token", param_hint=label.upper()
            )

    instance = Project.open(project)
    try:
        try:
            result = instance.corpus_collocation_evidence(
                scope_id,
                query,
                collocate,
                window=window,
                min_frequency=min_frequency,
            )
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise
        _emit(result)
    finally:
        instance.close()


@corpus_app.command("build-tokens")
def corpus_build_tokens(project: Path, revision: str) -> None:
    """Build a versioned Parquet/Zstd derived surface-token artifact."""
    instance = Project.open(project)
    try:
        _emit(instance.build_surface_token_artifact(revision).__dict__)
    finally:
        instance.close()


@corpus_app.command("build-nlp")
def corpus_build_nlp(
    project: Path,
    revision: str,
    language: str,
    backend: str = typer.Option(..., help="spacy or stanza"),
    model: str = typer.Option(..., help="Already installed local model identifier."),
    model_dir: Path | None = typer.Option(None, help="Required local model directory for Stanza."),
) -> None:
    """Build a local, versioned NLP Parquet artifact without downloading a model."""
    if backend == "spacy":
        selected = load_spacy(language, model)
    elif backend == "stanza" and model_dir is not None:
        selected = load_stanza(language, model, str(model_dir))
    elif backend == "stanza":
        raise typer.BadParameter("--model-dir is required for Stanza", param_hint="--model-dir")
    else:
        raise typer.BadParameter("backend must be spacy or stanza", param_hint="--backend")
    instance = Project.open(project)
    try:
        _emit(instance.build_parsed_token_artifact(revision, selected).__dict__)
    finally:
        instance.close()


@corpus_app.command("create")
def corpus_create(project: Path, name: str, description: str | None = typer.Option(None)) -> None:
    """Create an empty named corpus; documents are added by immutable revision ID."""
    instance = Project.open(project)
    try:
        _emit({"corpus_id": instance.create_corpus(name, description), "name": name})
    finally:
        instance.close()


@corpus_app.command("add")
def corpus_add(project: Path, corpus: str, revision: str) -> None:
    """Add one immutable document revision to a named corpus."""
    instance = Project.open(project)
    try:
        instance.add_revision_to_corpus(corpus, revision)
        _emit({"corpus_id": corpus, "revision_id": revision, "status": "added"})
    finally:
        instance.close()


@corpus_app.command("snapshot")
def corpus_snapshot(
    project: Path,
    corpus: str,
    output_format: str = typer.Option("json", "--format", help="json or text."),
) -> None:
    """Freeze a named corpus membership as a reusable immutable scope snapshot."""
    if output_format not in {"json", "text"}:
        raise typer.BadParameter("format must be json or text", param_hint="--format")
    instance = Project.open(project)
    try:
        scope_id = instance.corpus_snapshot(corpus)
        revision_ids = instance.scope_revisions(scope_id)
        payload = {
            "corpus_id": corpus,
            "scope_id": scope_id,
            "revision_count": len(revision_ids),
            "revision_ids": revision_ids,
        }
        if output_format == "json":
            _emit(payload)
        else:
            typer.echo(f"Scope snapshot: {scope_id}")
            typer.echo(f"Corpus: {corpus}")
            typer.echo(f"Revisions: {len(revision_ids)}")
            for revision_id in revision_ids:
                typer.echo(f"  {revision_id}")
    finally:
        instance.close()


@corpus_app.command("snapshot-subcorpus")
def corpus_snapshot_subcorpus(project: Path, subcorpus_id: str) -> None:
    """Freeze a saved subcorpus membership as a reusable immutable scope snapshot."""
    if not subcorpus_id.strip() or subcorpus_id != subcorpus_id.strip():
        raise typer.BadParameter(
            "subcorpus ID must be non-empty and have no surrounding whitespace",
            param_hint="SUBCORPUS_ID",
        )

    instance = Project.open(project)
    try:
        scope_id = instance.subcorpus_snapshot(subcorpus_id)
        revision_ids = instance.scope_revisions(scope_id)
        _emit(
            {
                "subcorpus_id": subcorpus_id,
                "scope_id": scope_id,
                "revision_count": len(revision_ids),
                "revision_ids": revision_ids,
            }
        )
    finally:
        instance.close()


@corpus_app.command("scope-revisions")
def corpus_scope_revisions(
    project: Path,
    scope_id: str,
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for the immutable scope membership page.",
    ),
    limit: int = typer.Option(
        MAX_SCOPE_REVISION_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_SCOPE_REVISION_PAGE_SIZE,
        help="Maximum revision IDs in this page (1..100).",
    ),
) -> None:
    """Read a bounded page of revision IDs from an existing ScopeSnapshot."""
    instance = Project.open(project, read_only=True)
    try:
        _emit(instance.scope_revisions_page(scope_id, offset=offset, limit=limit))
    finally:
        instance.close()


@corpus_app.command("list")
def corpus_list(
    project: Path,
    offset: int | None = typer.Option(
        None,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for a bounded corpus page.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum corpora in this page (1..1000).",
    ),
) -> None:
    """List corpus memberships without expanding any analysis scope."""
    instance = Project.open(project)
    try:
        if offset is None and limit is None:
            _emit([dict(row) for row in instance.list_corpora()])
        else:
            _emit(
                instance.corpora_page(
                    offset=0 if offset is None else offset,
                    limit=100 if limit is None else limit,
                )
            )
    finally:
        instance.close()


@corpus_app.command("create-subcorpus")
def corpus_create_subcorpus(
    project: Path,
    corpus: str,
    name: str,
    revisions: str = typer.Option(
        ..., help="Comma-separated revision IDs already in the parent corpus."
    ),
    description: str | None = typer.Option(None),
) -> None:
    """Freeze an explicit named subset of one corpus for reproducible comparison."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        subcorpus_id = instance.create_subcorpus(corpus, name, revision_ids, description)
        _emit({"subcorpus_id": subcorpus_id, "corpus_id": corpus, "name": name})
    finally:
        instance.close()


@corpus_app.command("list-subcorpora")
def corpus_list_subcorpora(
    project: Path,
    corpus: str | None = typer.Option(None),
    offset: int | None = typer.Option(
        None,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for a bounded subcorpus page.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum subcorpora in this page (1..1000).",
    ),
) -> None:
    """List explicit frozen subcorpora without expanding their memberships."""
    instance = Project.open(project)
    try:
        if offset is None and limit is None:
            _emit([dict(row) for row in instance.list_subcorpora(corpus)])
        else:
            _emit(
                instance.subcorpora_page(
                    corpus_id=corpus or None,
                    offset=0 if offset is None else offset,
                    limit=100 if limit is None else limit,
                )
            )
    finally:
        instance.close()


@corpus_app.command("normalized-frequency")
def corpus_normalized_frequency(
    project: Path,
    revision: str,
    artifacts: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, help="Zero-based token-type offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum rows in this page."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return a bounded frequency page from explicit derived token artifacts."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    artifact_ids = [artifact.strip() for artifact in artifacts.split(",") if artifact.strip()]
    instance = Project.open(project)
    try:
        scope_id = instance.snapshot([revision])
        page = instance.corpus_surface_frequency_page(
            scope_id,
            artifact_ids,
            field="normalized",
            offset=offset,
            limit=limit,
        )
        _emit_counter_page(
            dict(page),
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
        )
    finally:
        instance.close()


@corpus_app.command("document-frequency")
def corpus_document_frequency(
    project: Path,
    revisions: str,
    artifacts: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, help="Zero-based token-type offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum rows in this page."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return a bounded document-frequency page for an explicit revision scope."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    revision_ids = [revision.strip() for revision in revisions.split(",") if revision.strip()]
    artifact_ids = [artifact.strip() for artifact in artifacts.split(",") if artifact.strip()]
    instance = Project.open(project)
    try:
        scope_id = instance.snapshot(revision_ids)
        page = instance.corpus_surface_frequency_page(
            scope_id,
            artifact_ids,
            field="document_frequency",
            offset=offset,
            limit=limit,
        )
        _emit_counter_page(
            dict(page),
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
        )
    finally:
        instance.close()


@corpus_app.command("parsed-frequency")
def corpus_parsed_frequency(
    project: Path,
    revisions: str = typer.Option(..., help="Comma-separated immutable revision IDs."),
    artifacts: str = typer.Option(..., help="Comma-separated parsed-token artifact IDs."),
    field: str = typer.Option(..., help="lemma, pos, or dependency_relation."),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, help="Zero-based value offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum rows in this page."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return a bounded parser-derived frequency page for an immutable scope."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    artifact_ids = [item.strip() for item in artifacts.split(",") if item.strip()]
    if field not in {"lemma", "pos", "dependency_relation"}:
        raise typer.BadParameter(
            "field must be lemma, pos, or dependency_relation", param_hint="--field"
        )
    instance = Project.open(project)
    try:
        scope_id = instance.snapshot(revision_ids)
        page = instance.corpus_parsed_frequency_page(
            scope_id,
            artifact_ids,
            field=field,
            offset=offset,
            limit=limit,
        )
        page_rows = page.get("rows")
        if not isinstance(page_rows, list) or any(
            not isinstance(row, dict) or "value" not in row or "count" not in row
            for row in page_rows
        ):
            raise typer.BadParameter("frequency page rows are invalid")
        # Retain the established CLI row key while Core uses the more general
        # ``value`` field for parsed annotations.
        rows = cast(list[dict[str, object]], page_rows)
        _emit_counter_page(
            {
                **page,
                "rows": [{"token": row["value"], "count": row["count"]} for row in rows],
            },
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
        )
    finally:
        instance.close()


@corpus_app.command("parsed-frequency-scope")
def corpus_parsed_frequency_scope(
    project: Path,
    scope_id: str,
    artifacts: str = typer.Option(
        ..., "--artifacts", help="Comma-separated parsed-token artifact IDs for the scope."
    ),
    field: str = typer.Option(..., "--field", help="lemma, pos, or dependency_relation."),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based value offset.",
    ),
    limit: int = typer.Option(100, "--limit", min=1, max=1_000),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Return a bounded parser-derived frequency page from an existing scope."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    artifact_ids = [item.strip() for item in artifacts.split(",") if item.strip()]
    if not artifact_ids:
        raise typer.BadParameter(
            "--artifacts must include at least one parsed-token artifact ID",
            param_hint="--artifacts",
        )
    if len(artifact_ids) != len(set(artifact_ids)):
        raise typer.BadParameter("--artifacts must not contain duplicates", param_hint="--artifacts")
    if field not in {"lemma", "pos", "dependency_relation"}:
        raise typer.BadParameter(
            "field must be lemma, pos, or dependency_relation", param_hint="--field"
        )

    instance = Project.open(project, read_only=True)
    try:
        try:
            page = instance.corpus_parsed_frequency_page(
                scope_id,
                artifact_ids,
                field=field,
                offset=offset,
                limit=limit,
            )
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise
        page_rows = page.get("rows")
        if not isinstance(page_rows, list) or any(
            not isinstance(row, dict) or "value" not in row or "count" not in row
            for row in page_rows
        ):
            raise typer.BadParameter("frequency page rows are invalid")
        rows = cast(list[dict[str, object]], page_rows)
        _emit_counter_page(
            {
                **page,
                "rows": [{"token": row["value"], "count": row["count"]} for row in rows],
            },
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
        )
    finally:
        instance.close()


@corpus_app.command("lemma-kwic")
def corpus_lemma_kwic(
    project: Path,
    lemma: str,
    revisions: str = typer.Option(
        ..., "--revisions", help="Comma-separated immutable revision IDs."
    ),
    artifacts: str = typer.Option(
        ...,
        "--artifacts",
        help="Comma-separated parsed-token artifact IDs, one for each revision.",
    ),
    window: int = typer.Option(
        5, min=0, max=100, help="Number of parsed tokens to show on either side."
    ),
    offset: int = typer.Option(0, min=0, help="Zero-based result offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum rows in this page."),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Show lemma matches from explicitly selected revisions and parsed artifacts."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    artifact_ids = [item.strip() for item in artifacts.split(",") if item.strip()]
    if not revision_ids:
        raise typer.BadParameter(
            "--revisions must include at least one revision ID", param_hint="--revisions"
        )
    if not artifact_ids:
        raise typer.BadParameter(
            "--artifacts must include at least one parsed-token artifact ID",
            param_hint="--artifacts",
        )

    instance = Project.open(project)
    try:
        scope_id = instance.snapshot(revision_ids)
        page = instance.corpus_lemma_kwic(
            scope_id,
            artifact_ids,
            lemma,
            window=window,
            offset=offset,
            limit=limit,
        )
        if output_format == "json":
            _emit(page)
        else:
            rows = page["rows"]
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise typer.BadParameter("lemma-KWIC page rows are invalid")
            _emit_rows(
                cast(list[Mapping[str, object]], rows),
                output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
            )
            page_offset = page["offset"]
            total_count = page["total_count"]
            if not isinstance(page_offset, int) or not isinstance(total_count, int):
                raise typer.BadParameter("lemma-KWIC page metadata is invalid")
            if rows:
                status = (
                    f"lemma-KWIC rows {page_offset + 1}–{page_offset + len(rows)} of {total_count}"
                )
            else:
                status = f"no lemma-KWIC rows at offset {page_offset}; total={total_count}"
            typer.echo(f"{status}; has_more={str(page['has_more']).lower()}", err=True)
    finally:
        instance.close()


@corpus_app.command("lemma-kwic-evidence")
def corpus_lemma_kwic_evidence(
    project: Path,
    lemma: str,
    revisions: str = typer.Option(
        ..., "--revisions", help="Comma-separated immutable revision IDs."
    ),
    artifacts: str = typer.Option(
        ...,
        "--artifacts",
        help="Comma-separated parsed-token artifact IDs, one for each revision.",
    ),
    revision_id: str = typer.Option(
        ..., "--revision-id", help="Revision ID containing the selected lemma occurrence."
    ),
    start_offset: int = typer.Option(
        ..., "--start-offset", min=0, help="Inclusive canonical character offset from a KWIC row."
    ),
    end_offset: int = typer.Option(
        ..., "--end-offset", min=1, help="Exclusive canonical character offset from a KWIC row."
    ),
) -> None:
    """Persist one selected lemma-KWIC occurrence as cited textual evidence."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    artifact_ids = [item.strip() for item in artifacts.split(",") if item.strip()]
    if not lemma.strip():
        raise typer.BadParameter("lemma must not be empty", param_hint="lemma")
    if not revision_ids:
        raise typer.BadParameter(
            "--revisions must include at least one revision ID", param_hint="--revisions"
        )
    if not artifact_ids:
        raise typer.BadParameter(
            "--artifacts must include at least one parsed-token artifact ID",
            param_hint="--artifacts",
        )
    if revision_id not in revision_ids:
        raise typer.BadParameter(
            "--revision-id must be included in --revisions", param_hint="--revision-id"
        )
    if end_offset <= start_offset:
        raise typer.BadParameter(
            "--end-offset must be greater than --start-offset", param_hint="--end-offset"
        )

    instance = Project.open(project)
    try:
        scope_id = instance.snapshot(revision_ids)
        evidence = instance.corpus_lemma_kwic_evidence(
            scope_id,
            artifact_ids,
            revision_id,
            lemma,
            start_offset,
            end_offset,
        )
        _emit(evidence)
    finally:
        instance.close()


@corpus_app.command("lemma-kwic-evidence-scope")
def corpus_lemma_kwic_evidence_scope(
    project: Path,
    scope_id: str,
    lemma: str,
    artifacts: str = typer.Option(
        ...,
        "--artifacts",
        help="Comma-separated parsed-token artifact IDs covering the existing scope.",
    ),
    revision_id: str = typer.Option(
        ..., "--revision-id", help="Revision ID containing the selected lemma occurrence."
    ),
    start_offset: int = typer.Option(
        ..., "--start-offset", min=0, help="Inclusive canonical character offset from a KWIC row."
    ),
    end_offset: int = typer.Option(
        ..., "--end-offset", min=1, help="Exclusive canonical character offset from a KWIC row."
    ),
) -> None:
    """Persist one selected lemma-KWIC occurrence in an existing immutable scope."""
    artifact_ids = [item.strip() for item in artifacts.split(",") if item.strip()]
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    if not lemma.strip():
        raise typer.BadParameter("lemma must not be empty", param_hint="LEMMA")
    if len(lemma) > 256:
        raise typer.BadParameter("lemma must be at most 256 characters", param_hint="LEMMA")
    if not artifact_ids:
        raise typer.BadParameter(
            "--artifacts must include at least one parsed-token artifact ID",
            param_hint="--artifacts",
        )
    if not revision_id.strip():
        raise typer.BadParameter("revision ID cannot be blank", param_hint="--revision-id")
    if end_offset <= start_offset:
        raise typer.BadParameter(
            "--end-offset must be greater than --start-offset", param_hint="--end-offset"
        )

    instance = Project.open(project)
    try:
        evidence = instance.corpus_lemma_kwic_evidence(
            scope_id,
            artifact_ids,
            revision_id,
            lemma,
            start_offset,
            end_offset,
        )
        _emit(evidence)
    finally:
        instance.close()


@corpus_app.command("dependency-patterns")
def corpus_dependency_patterns(
    project: Path,
    revisions: str = typer.Option(..., help="Comma-separated immutable revision IDs."),
    artifacts: str = typer.Option(..., help="One parsed-token artifact ID per revision."),
    relation: str | None = typer.Option(None, "--relation"),
    head_lemma: str | None = typer.Option(None, "--head-lemma"),
    dependent_lemma: str | None = typer.Option(None, "--dependent-lemma"),
    offset: int = typer.Option(0, "--offset", min=0, help="Zero-based result offset."),
    limit: int = typer.Option(100, "--limit", min=1, max=1_000),
) -> None:
    """Aggregate syntactic head→dependent candidates from complete scoped parses."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    artifact_ids = [item.strip() for item in artifacts.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        scope_id = instance.snapshot(revision_ids)
        _emit(
            dependency_patterns(
                instance,
                scope_id,
                artifact_ids,
                relation=relation,
                head_lemma=head_lemma,
                dependent_lemma=dependent_lemma,
                offset=offset,
                limit=limit,
            )
        )
    finally:
        instance.close()


@corpus_app.command("dependency-patterns-scope")
def corpus_dependency_patterns_scope(
    project: Path,
    scope_id: str,
    artifacts: str = typer.Option(
        ..., "--artifacts", help="Comma-separated parsed-token artifact IDs for the scope."
    ),
    relation: str | None = typer.Option(None, "--relation"),
    head_lemma: str | None = typer.Option(None, "--head-lemma"),
    dependent_lemma: str | None = typer.Option(None, "--dependent-lemma"),
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based result offset.",
    ),
    limit: int = typer.Option(100, "--limit", min=1, max=1_000),
) -> None:
    """Aggregate dependency-pattern candidates from an existing immutable scope."""
    artifact_ids = [item.strip() for item in artifacts.split(",") if item.strip()]
    if not scope_id.strip() or scope_id != scope_id.strip() or len(scope_id) > 128:
        raise typer.BadParameter(
            "scope ID must be trimmed and contain 1..128 characters", param_hint="SCOPE_ID"
        )
    if not 1 <= len(artifact_ids) <= 10_000:
        raise typer.BadParameter(
            "--artifacts must include 1..10000 parsed-token artifact IDs",
            param_hint="--artifacts",
        )
    if any(not item.strip() or item != item.strip() or len(item) > 128 for item in artifact_ids):
        raise typer.BadParameter(
            "--artifacts IDs must be trimmed, nonblank, and at most 128 characters",
            param_hint="--artifacts",
        )
    if len(artifact_ids) != len(set(artifact_ids)):
        raise typer.BadParameter("--artifacts must not contain duplicates", param_hint="--artifacts")
    for value, option, label in (
        (relation, "--relation", "relation"),
        (head_lemma, "--head-lemma", "head lemma"),
        (dependent_lemma, "--dependent-lemma", "dependent lemma"),
    ):
        if value is not None and (not value.strip() or len(value) > 256):
            raise typer.BadParameter(
                f"{label} must be nonblank and at most 256 characters", param_hint=option
            )

    instance = Project.open(project, read_only=True)
    try:
        try:
            result = instance.corpus_dependency_patterns(
                scope_id,
                artifact_ids,
                relation=relation,
                head_lemma=head_lemma,
                dependent_lemma=dependent_lemma,
                offset=offset,
                limit=limit,
            )
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="SCOPE_ID") from None
            raise
        _emit(result)
    finally:
        instance.close()


@corpus_app.command("compare-frequency")
def corpus_compare_frequency(
    project: Path,
    left_revision: str,
    right_revision: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, help="Ranked result offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum ranked rows to return."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Page normalized surface-token rates between two explicit revisions."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        result = compare_frequency_page(
            instance,
            instance.snapshot([left_revision]),
            instance.snapshot([right_revision]),
            offset=offset,
            limit=limit,
        )
        if output_format == "json":
            _emit(result)
        else:
            _emit_comparison_page_rows(
                result,
                label="frequency",
                output_format=output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
            )
    finally:
        instance.close()


@corpus_app.command("keyness")
def corpus_keyness(
    project: Path,
    left_revision: str,
    right_revision: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, help="Ranked result offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum ranked rows to return."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Page deterministic signed log-likelihood keyness across explicit scopes."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        result = keyness_page(
            instance,
            instance.snapshot([left_revision]),
            instance.snapshot([right_revision]),
            offset=offset,
            limit=limit,
        )
        if output_format == "json":
            _emit(result)
        else:
            _emit_comparison_page_rows(
                result,
                label="keyness",
                output_format=output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
            )
    finally:
        instance.close()


@corpus_app.command("keyness-scopes")
def corpus_keyness_scopes(
    project: Path,
    left_scope_id: str,
    right_scope_id: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=1_000_000,
        help="Ranked result offset.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum ranked rows to return (1..1000).",
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Compare two existing scopes using deterministic signed log-likelihood keyness."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if not left_scope_id.strip():
        raise typer.BadParameter("left scope ID cannot be blank", param_hint="LEFT_SCOPE_ID")
    if not right_scope_id.strip():
        raise typer.BadParameter("right scope ID cannot be blank", param_hint="RIGHT_SCOPE_ID")

    instance = Project.open(project, read_only=True)
    try:
        try:
            result = instance.corpus_keyness_page(
                left_scope_id,
                right_scope_id,
                offset=offset,
                limit=limit,
            )
        except ScopeViolationError as exc:
            if str(exc) == "scope is required":
                raise typer.BadParameter("unknown scope", param_hint="LEFT_SCOPE_ID/RIGHT_SCOPE_ID") from None
            raise
        if output_format == "json":
            payload: dict[str, object] = dict(result)
            payload["left_scope_id"] = left_scope_id
            payload["right_scope_id"] = right_scope_id
            _emit(payload)
        else:
            _emit_comparison_page_rows(
                result,
                label="keyness",
                output_format=output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
                empty_csv_columns=("token", "left_count", "right_count", "log_likelihood"),
                status_context=(
                    f"left_scope_id={left_scope_id}; right_scope_id={right_scope_id}"
                ),
            )
    finally:
        instance.close()


@corpus_app.command("compare-named")
def corpus_compare_named(
    project: Path,
    left_corpus: str,
    right_corpus: str,
    metric: str = typer.Option("frequency", help="frequency or keyness"),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(
        0, min=0, help="Ranked row offset; use the comparison report for later pages."
    ),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum ranked rows to return."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Compare two named corpora through immutable membership snapshots."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        result = compare_named_corpora(
            instance, left_corpus, right_corpus, metric=metric, offset=offset, limit=limit
        )
        if output_format == "json":
            _emit(result)
        else:
            provenance_keys = (
                "comparison_id",
                "metric",
                "calculation_version",
                "left_corpus_id",
                "right_corpus_id",
                "left_scope_id",
                "right_scope_id",
                "offset",
                "limit",
                "total_rows",
                "has_more",
            )
            provenance = {key: result[key] for key in provenance_keys}
            rows = cast(list[Mapping[str, object]], result["rows"])
            _emit_rows(
                [{**provenance, **row} for row in rows],
                output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
            )
    finally:
        instance.close()


@corpus_app.command("compare-documents")
def corpus_compare_documents(
    project: Path,
    left_revision: str,
    right_revision: str,
    metric: str = typer.Option("frequency", help="frequency or keyness"),
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, help="Ranked row offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum ranked rows to return."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Persist a comparison of two immutable document revisions with both scopes visible."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        result = compare_documents(
            instance,
            left_revision,
            right_revision,
            metric=metric,
            offset=offset,
            limit=limit,
        )
        if output_format == "json":
            _emit(result)
        else:
            _emit_rows(
                cast(list[Mapping[str, object]], result["rows"]),
                output_format,
                spreadsheet_safe_csv=spreadsheet_safe_csv,
            )
    finally:
        instance.close()


@corpus_app.command("comparison-finding")
def corpus_comparison_finding(
    project: Path,
    comparison_run: str,
    token: str,
    comparability_basis: str = typer.Option(
        ...,
        "--comparability-basis",
        help="Researcher-stated reason these populations can be compared.",
    ),
) -> None:
    """Create a validated deterministic Finding from one persisted comparison row."""
    instance = Project.open(project)
    try:
        _emit(
            instance.create_comparative_finding(
                comparison_run, token, comparability_basis=comparability_basis
            )
        )
    finally:
        instance.close()


@coding_app.command("create-codebook")
def coding_create_codebook(project: Path, name: str, description: str | None = None) -> None:
    """Create a local qualitative codebook."""
    instance = Project.open(project)
    try:
        _emit({"codebook_id": instance.create_codebook(name, description)})
    finally:
        instance.close()


@coding_app.command("list-codebooks")
def coding_list_codebooks(
    project: Path,
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for the codebook page.",
    ),
    limit: int = typer.Option(
        MAX_CODEBOOK_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_CODEBOOK_PAGE_SIZE,
        help="Maximum codebooks in this page (1..100).",
    ),
) -> None:
    """Read one bounded page of project codebooks."""
    instance = Project.open(project, read_only=True)
    try:
        _emit(instance.codebook_pages(offset=offset, limit=limit))
    finally:
        instance.close()


@coding_app.command("list-codes")
def coding_list_codes(
    project: Path,
    codebook: str,
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for the code page.",
    ),
    limit: int = typer.Option(
        MAX_CODEBOOK_PAGE_SIZE,
        "--limit",
        min=1,
        max=MAX_CODEBOOK_PAGE_SIZE,
        help="Maximum codes in this page (1..100).",
    ),
) -> None:
    """Read one bounded page of codes, including unused codes and parent links."""
    instance = Project.open(project, read_only=True)
    try:
        _emit(instance.codes_for_codebook_page(codebook, offset=offset, limit=limit))
    finally:
        instance.close()


@coding_app.command("create-code")
def coding_create_code(
    project: Path,
    codebook: str,
    label: str,
    definition: str | None = typer.Option(
        None, help="Optional researcher-defined code definition."
    ),
    parent: str | None = typer.Option(None, help="Optional parent code in the same codebook."),
) -> None:
    """Create a controlled qualitative code."""
    instance = Project.open(project)
    try:
        _emit(
            {
                "code_id": instance.create_code(
                    codebook, label, definition=definition, parent_code_id=parent
                )
            }
        )
    finally:
        instance.close()


@coding_app.command("annotate")
def coding_annotate(
    project: Path,
    code: str,
    span: str,
    memo: str | None = typer.Option(None, help="Optional human coding memo."),
) -> None:
    """Attach one code to one canonical SourceSpan."""
    instance = Project.open(project)
    try:
        _emit({"annotation_id": instance.annotate_span(code, span, memo)})
    finally:
        instance.close()


@coding_app.command("annotations")
def coding_annotations(
    project: Path,
    revisions: str,
    offset: int | None = typer.Option(
        None,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for a bounded annotation page.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        max=100,
        help="Maximum annotations in this page (1..100).",
    ),
) -> None:
    """List coding annotations only for an explicit immutable revision scope."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        if offset is None and limit is None:
            items: list[dict[str, object]] = []
            page_offset = 0
            while True:
                page = instance.annotation_pages_for_revisions(
                    revision_ids, offset=page_offset, limit=100
                )
                page_items = cast(list[dict[str, object]], page["items"])
                items.extend(page_items)
                if not cast(bool, page["has_more"]):
                    break
                page_offset += len(page_items)
            _emit(items)
        else:
            _emit(
                instance.annotation_pages_for_revisions(
                    revision_ids,
                    offset=0 if offset is None else offset,
                    limit=100 if limit is None else limit,
                )
            )
    finally:
        instance.close()


@coding_app.command("create-memo")
def coding_create_memo(project: Path, revisions: str, title: str, content: str) -> None:
    """Persist a researcher memo against an explicit frozen revision selection."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        scope = instance.snapshot(revision_ids)
        _emit({"memo_id": instance.create_research_memo(scope, title, content), "scope_id": scope})
    finally:
        instance.close()


@coding_app.command("memos")
def coding_memos(
    project: Path,
    scope: str,
    offset: int | None = typer.Option(
        None,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based offset for a bounded memo page.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        max=MAX_RESEARCH_MEMO_PAGE_SIZE,
        help="Maximum memos in this page (1..100).",
    ),
) -> None:
    """List research memos stored for one immutable scope snapshot."""
    instance = Project.open(project)
    try:
        if offset is None and limit is None:
            _emit([dict(item) for item in instance.research_memos(scope)])
        else:
            _emit(
                instance.research_memos_page(
                    scope,
                    offset=0 if offset is None else offset,
                    limit=MAX_RESEARCH_MEMO_PAGE_SIZE if limit is None else limit,
                )
            )
    finally:
        instance.close()


@nlp_app.command("capabilities")
def nlp_capabilities(
    language: str,
    backend: str | None = typer.Option(
        None, "--backend", help="Explicit local backend: spacy or stanza."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Already-installed local model identifier."
    ),
    model_dir: Path | None = typer.Option(
        None, "--model-dir", help="Required local Stanza resources directory."
    ),
) -> None:
    """Report local preprocessing capabilities; no model is downloaded or substituted."""
    if backend is None and model is None and model_dir is None:
        capabilities = local_language_capabilities(language)
    elif backend is None or model is None:
        raise typer.BadParameter("--backend and --model must be supplied together")
    else:
        try:
            capabilities = inspect_local_backend(
                language,
                backend=backend,
                model=model,
                model_dir=str(model_dir) if model_dir is not None else None,
            )
        except ValueError as exc:
            raise typer.BadParameter(_safe_project_error_message(exc)) from None
    _emit([item.__dict__ for item in capabilities])


@profile_app.command("save-controlled")
def profile_save_controlled(project: Path, name: str, profile_file: Path) -> None:
    """Save a JSON declarative controlled-CDA profile; arbitrary executors are rejected."""
    try:
        payload = json.loads(profile_file.read_text(encoding="utf-8"))
        profile = ControlledCdaProfile.model_validate(payload)
        profile.validate_method_contract()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise typer.BadParameter(
            "profile file must contain a valid controlled-CDA JSON object"
        ) from exc
    instance = Project.open(project)
    try:
        _emit(
            {"profile_id": instance.save_analysis_profile(name, profile.model_dump()), "name": name}
        )
    finally:
        instance.close()


@profile_app.command("run")
def profile_run(project: Path, name: str, revisions: str, preset: str | None = None) -> None:
    """Run a saved profile only through the trusted controlled CDA executor."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        _emit(run_saved_controlled_profile(instance, revision_ids, name, preset_name=preset))
    finally:
        instance.close()


@profile_app.command("batch")
def profile_batch(
    project: Path,
    name: str,
    revisions: str,
    preset: str | None = typer.Option(
        None, "--preset", help="Override the profile's default preset."
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error",
        help="Keep analyzing selected revisions after a revision fails and report each status.",
    ),
) -> None:
    """Run a saved profile once per document revision and emit a JSON result array."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        results = run_batch_saved_controlled_profile(
            instance,
            revision_ids,
            name,
            preset_name=preset,
            continue_on_error=continue_on_error,
        )
        _emit(results)
        if continue_on_error and any(
            isinstance(result, dict) and result.get("status") == "failed" for result in results
        ):
            raise typer.Exit(code=1)
    finally:
        instance.close()


@profile_app.command("list")
def profile_list(
    project: Path,
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, jsonl, or csv."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List stored declarative profiles and their immutable declared versions."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        _emit_records(
            instance.list_analysis_profiles(),
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
        )
    finally:
        instance.close()


@profile_app.command("info")
def profile_info(project: Path, name: str) -> None:
    """Read one stored declarative profile without executing it."""
    instance = Project.open(project)
    try:
        _emit(instance.analysis_profile(name))
    finally:
        instance.close()


@skills_app.command("list")
def skills_list(
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, jsonl, or csv."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List trusted built-in skills and their execution contracts."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    rows = [
        {
            "id": manifest.id,
            "version": manifest.version,
            "framework": manifest.framework,
            "execution_kind": manifest.execution_kind,
            "required_capabilities": ", ".join(manifest.required_capabilities) or "—",
        }
        for manifest in BUILTIN_SKILLS.manifests()
    ]
    _emit_records(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)


@skills_app.command("info")
def skills_info(skill_id: str) -> None:
    """Show a machine-readable manifest for exactly one trusted built-in skill."""
    _emit(builtin_manifest(skill_id).model_dump())


@skills_app.command("docs")
def skills_docs(skill_id: str) -> None:
    """Print the bundled methodology documentation for one built-in skill."""
    typer.echo(skill_documentation(skill_id), nl=False)


@skills_app.command("quote-attributions")
def skills_quote_attributions(
    project: Path,
    revision_ids: list[str] = typer.Option(
        ...,
        "--revision",
        help="One selected immutable document revision; repeat to include more revisions.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help=(
            "Language: en or es. Assign it to every selected revision for reviewed cue scans. "
            "Without it, quote marks are inventoried without inferring attribution cues."
        ),
    ),
    output_format: str = typer.Option(
        "json", "--format", help="Output format: json, jsonl, csv, or table."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Inventory quote-attribution candidates in only the selected revisions."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if language is not None and language not in {"en", "es"}:
        raise typer.BadParameter("language must be en or es", param_hint="--language")
    if any(not revision_id.strip() for revision_id in revision_ids):
        raise typer.BadParameter("revision IDs must not be empty", param_hint="--revision")
    if len(set(revision_ids)) != len(revision_ids):
        raise typer.BadParameter("revision IDs must not be duplicated", param_hint="--revision")

    revision_languages = (
        {revision_id: language for revision_id in revision_ids} if language is not None else None
    )
    instance = Project.open(project)
    try:
        result = run_quote_attribution_candidate_analysis(
            instance,
            revision_ids,
            revision_languages=revision_languages,
        )
        if output_format == "json":
            _emit(result)
            return

        candidates = result.get("candidates")
        if not isinstance(candidates, list) or any(
            not isinstance(candidate, Mapping) for candidate in candidates
        ):
            raise ProjectError("quote-attribution candidate result is invalid")
        _emit_rows(
            cast(list[Mapping[str, object]], candidates),
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
        )
        typer.echo(
            "quote-attribution candidates: "
            f"analysis_run_id={result.get('analysis_run_id')}; "
            f"scope_snapshot_id={result.get('scope_snapshot_id')}; "
            f"candidate_count={result.get('candidate_count')}; "
            f"attribution_candidate_count={result.get('attribution_candidate_count')}; "
            f"unmatched_marker_count={result.get('unmatched_marker_count')}; "
            f"ambiguous_marker_count={result.get('ambiguous_marker_count')}",
            err=True,
        )
    finally:
        instance.close()


def _scope_question_result_payload(result: object) -> dict[str, object]:
    """Validate and JSON-normalize the standalone scope-question result."""
    if not isinstance(result, Mapping):
        raise ProjectError("scope-question candidate result is invalid")
    payload = cast(dict[str, object], dict(result))
    required = {
        "analysis_run_id",
        "scope_snapshot_id",
        "skill_run_id",
        "retrieval_run_id",
        "finding_id",
        "evidence_ids",
        "coverage_id",
        "candidate_count",
        "coverage",
    }
    if not required.issubset(payload):
        raise ProjectError("scope-question candidate result is invalid")

    for field in (
        "analysis_run_id",
        "scope_snapshot_id",
        "skill_run_id",
        "retrieval_run_id",
        "coverage_id",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ProjectError("scope-question candidate result is invalid")
    if payload["finding_id"] is not None and not isinstance(payload["finding_id"], str):
        raise ProjectError("scope-question candidate result is invalid")

    evidence_ids = payload["evidence_ids"]
    candidate_count = payload["candidate_count"]
    if (
        not isinstance(evidence_ids, list)
        or any(not isinstance(evidence_id, str) for evidence_id in evidence_ids)
        or type(candidate_count) is not int
        or candidate_count < 0
        or candidate_count != len(evidence_ids)
    ):
        raise ProjectError("scope-question candidate result is invalid")

    coverage = payload["coverage"]
    if isinstance(coverage, Mapping):
        coverage_payload = cast(dict[str, object], dict(coverage))
    else:
        model_dump = getattr(coverage, "model_dump", None)
        if not callable(model_dump):
            raise ProjectError("scope-question candidate result is invalid")
        dumped_coverage = model_dump(mode="json")
        if not isinstance(dumped_coverage, Mapping):
            raise ProjectError("scope-question candidate result is invalid")
        coverage_payload = cast(dict[str, object], dict(dumped_coverage))
    if coverage_payload.get("coverage_mode") != "TARGETED":
        raise ProjectError("scope-question candidates must use TARGETED coverage")
    payload["coverage"] = coverage_payload
    return payload


def _scope_question_summary_row(payload: Mapping[str, object]) -> dict[str, object]:
    """Expose provenance and coverage without fabricating passage rows."""
    coverage = payload["coverage"]
    if not isinstance(coverage, Mapping):
        raise ProjectError("scope-question candidate result is invalid")
    return {
        "analysis_run_id": payload["analysis_run_id"],
        "scope_snapshot_id": payload["scope_snapshot_id"],
        "skill_run_id": payload["skill_run_id"],
        "retrieval_run_id": payload["retrieval_run_id"],
        "finding_id": payload["finding_id"],
        "coverage_id": payload["coverage_id"],
        "candidate_count": payload["candidate_count"],
        "coverage_mode": coverage.get("coverage_mode"),
        "negative_result_status": coverage.get("negative_result_status"),
    }


@skills_app.command("scope-question-candidates")
def skills_scope_question_candidates(
    project: Path,
    revision_ids: list[str] = typer.Option(
        ...,
        "--revision",
        help="One selected immutable document revision; repeat to include more revisions.",
    ),
    framework_id: str = typer.Option(
        ..., "--framework", help="Selected interpretive framework to target with the question."
    ),
    question: str = typer.Option(..., "--question", help="Researcher-authored retrieval question."),
    limit: int = typer.Option(
        MAX_FRAMEWORK_HITS,
        min=1,
        max=MAX_FRAMEWORK_HITS,
        help=f"Maximum lexical candidates to return (1–{MAX_FRAMEWORK_HITS}).",
    ),
    output_format: str = typer.Option(
        "json", "--format", help="Output format: json, jsonl, csv, or table."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Retrieve targeted lexical examples for a question in selected revisions."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if any(
        not revision_id.strip() or revision_id != revision_id.strip()
        for revision_id in revision_ids
    ):
        raise typer.BadParameter(
            "revision IDs must be nonempty and must not have surrounding whitespace",
            param_hint="--revision",
        )
    if len(set(revision_ids)) != len(revision_ids):
        raise typer.BadParameter("revision IDs must not be duplicated", param_hint="--revision")
    if framework_id not in FRAMEWORK_REVIEWS or framework_id == "critical-reviewer":
        raise typer.BadParameter(
            "framework must be a selected interpretive framework", param_hint="--framework"
        )
    if not question.strip():
        raise typer.BadParameter("question must not be empty", param_hint="--question")

    instance = Project.open(project)
    try:
        result = instance.run_scope_question_candidates(
            revision_ids,
            framework_id,
            question,
            limit=limit,
        )
        payload = _scope_question_result_payload(result)
        if output_format == "json":
            _emit(payload)
            return

        _emit_rows(
            [_scope_question_summary_row(payload)],
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
        )
        coverage = cast(Mapping[str, object], payload["coverage"])
        summary = (
            "scope-question-candidates: "
            f"analysis_run_id={payload['analysis_run_id']}; "
            f"scope_snapshot_id={payload['scope_snapshot_id']}; "
            f"retrieval_run_id={payload['retrieval_run_id']}; "
            f"candidate_count={payload['candidate_count']}; coverage=TARGETED"
        )
        if payload["candidate_count"] == 0:
            summary += (
                "; no examples were found under this targeted lexical plan; "
                "zero results do not establish absence"
            )
        else:
            summary += "; candidates are targeted examples, not an exhaustive answer"
        if coverage.get("negative_result_status") not in {None, "NOT_APPLICABLE"}:
            summary += f"; negative_result_status={coverage['negative_result_status']}"
        typer.echo(summary, err=True)
    finally:
        instance.close()


def _van_dijk_comparison_rows(
    result: Mapping[str, object],
) -> list[Mapping[str, object]]:
    groups = result.get("groups")
    if not isinstance(groups, list) or any(not isinstance(group, Mapping) for group in groups):
        raise ProjectError("Van Dijk group comparison result is invalid")
    rows: list[Mapping[str, object]] = []
    for group in groups:
        label = group.get("label")
        code_counts = group.get("code_counts")
        annotations = group.get("annotations")
        if (
            not isinstance(label, str)
            or not isinstance(code_counts, list)
            or not isinstance(annotations, list)
            or any(not isinstance(annotation, Mapping) for annotation in annotations)
        ):
            raise ProjectError("Van Dijk group comparison result is invalid")
        rows.append(
            {
                "row_type": "group",
                "group_label": label,
                "selected_annotation_count": group.get("selected_annotation_count"),
                "coding_status": group.get("coding_status"),
                "code_counts": render_json(code_counts),
            }
        )
        for annotation in annotations:
            rows.append(
                {
                    "row_type": "annotation",
                    "group_label": label,
                    "annotation_id": annotation.get("annotation_id"),
                    "code_id": annotation.get("code_id"),
                    "code_label": annotation.get("code_label"),
                    "code_definition": annotation.get("code_definition"),
                    "revision_id": annotation.get("revision_id"),
                    "span_id": annotation.get("span_id"),
                    "start_offset": annotation.get("start_offset"),
                    "end_offset": annotation.get("end_offset"),
                    "text": annotation.get("text"),
                    "evidence_id": annotation.get("evidence_id"),
                    "citation": annotation.get("citation"),
                }
            )
    return rows


def _van_dijk_comparison_metadata(result: Mapping[str, object]) -> dict[str, object]:
    def safe_identifier(field: str) -> str:
        value = result.get(field)
        if (
            isinstance(value, str)
            and 0 < len(value) <= 128
            and all(character.isalnum() or character in "-_." for character in value)
        ):
            return value
        return "unavailable"

    selected_count = result.get("selected_annotation_count")
    if type(selected_count) is not int or selected_count < 0:
        selected_count = 0
    groups = result.get("groups")
    group_count = len(groups) if isinstance(groups, list) and len(groups) <= 2 else 0
    return {
        "analysis_run_id": safe_identifier("analysis_run_id"),
        "skill_run_id": safe_identifier("skill_run_id"),
        "finding_id": safe_identifier("finding_id"),
        "scope_snapshot_id": safe_identifier("scope_snapshot_id"),
        "selected_annotation_count": selected_count,
        "group_count": group_count,
    }


@skills_app.command("van-dijk-group-comparison")
def skills_van_dijk_group_comparison(
    project: Path,
    revision_ids: list[str] = typer.Option(
        ...,
        "--revision",
        help="One selected immutable document revision; repeat to include more revisions.",
    ),
    input_path: Path = typer.Option(
        ...,
        "--input",
        help="UTF-8 JSON file with the two researcher-labeled groups and annotation IDs.",
    ),
    output_format: str = typer.Option(
        "json", "--format", help="Output format: json, jsonl, csv, or table."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Prepare a descriptive Van Dijk two-group comparison from selected annotations."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if any(
        not revision_id.strip() or revision_id != revision_id.strip()
        for revision_id in revision_ids
    ):
        raise typer.BadParameter(
            "revision IDs must be nonempty and must not have surrounding whitespace",
            param_hint="--revision",
        )
    if len(set(revision_ids)) != len(revision_ids):
        raise typer.BadParameter("revision IDs must not be duplicated", param_hint="--revision")

    payload = _read_van_dijk_group_comparison_input(input_path)
    instance = Project.open(project)
    try:
        result = instance.prepare_van_dijk_group_comparison(
            revision_ids,
            groups=cast(list[VanDijkGroupComparisonGroupInput], payload["groups"]),
            shared_topic_event=cast(str, payload["shared_topic_event"]),
            comparability_basis=cast(str, payload["comparability_basis"]),
            codebook_id=cast(str, payload["codebook_id"]),
        )
        if output_format == "json":
            _emit(result)
            return
        if not isinstance(result, Mapping):
            raise ProjectError("Van Dijk group comparison result is invalid")
        _emit_rows(
            _van_dijk_comparison_rows(result),
            output_format,
            spreadsheet_safe_csv=spreadsheet_safe_csv,
        )
        typer.echo(
            "van-dijk-group-comparison metadata "
            + render_json(_van_dijk_comparison_metadata(result)),
            err=True,
        )
    finally:
        instance.close()


@search_app.command("exact")
def search_exact(
    project: Path,
    revision: str,
    phrase: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, help="Zero-based result offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum results in this page."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Find a literal phrase in one revision and return a bounded result page."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        scope = instance.snapshot([revision])
        page = instance.search_exact_page(phrase, scope, offset=offset, limit=limit)
        rows = [dict(row) for row in cast(list[sqlite3.Row], page["rows"])]
        total_count = cast(int, page["total_count"])
        has_more = cast(bool, page["has_more"])
        payload = {
            "scope_id": scope,
            "revision_id": revision,
            "offset": offset,
            "limit": limit,
            "total_count": total_count,
            "has_more": has_more,
            "rows": rows,
        }
        if output_format == "json":
            _emit(payload)
        else:
            _emit_rows(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
            page_rows = len(rows)
            if page_rows:
                status = f"Exact-search rows {offset + 1}–{offset + page_rows} of {total_count}"
            else:
                status = f"No exact-search rows at offset {offset}; total={total_count}"
            typer.echo(f"{status}; has_more={str(has_more).lower()}", err=True)
    finally:
        instance.close()


@search_app.command("exact-scope")
def search_exact_scope(
    project: Path,
    scope_id: str,
    phrase: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(
        0,
        "--offset",
        min=0,
        max=0x7FFF_FFFF_FFFF_FFFF,
        help="Zero-based result offset.",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        min=1,
        max=1_000,
        help="Maximum results in this page (1..1000).",
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Find a literal phrase inside an existing scope and return a bounded page."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if output_format not in {"json", "jsonl", "csv", "table"}:
        raise typer.BadParameter("format must be json, jsonl, csv, or table", param_hint="--format")
    if not scope_id.strip():
        raise typer.BadParameter("scope ID cannot be blank", param_hint="SCOPE_ID")
    if not phrase.strip() or len(phrase) > 2_000:
        raise typer.BadParameter("phrase must contain 1..2000 characters", param_hint="PHRASE")

    instance = Project.open(project, read_only=True)
    try:
        page = instance.search_exact_page(phrase, scope_id, offset=offset, limit=limit)
        rows = [dict(row) for row in cast(list[sqlite3.Row], page["rows"])]
        total_count = cast(int, page["total_count"])
        has_more = cast(bool, page["has_more"])
        payload = {
            "scope_id": scope_id,
            "offset": offset,
            "limit": limit,
            "total_count": total_count,
            "has_more": has_more,
            "rows": rows,
        }
        if output_format == "json":
            _emit(payload)
        else:
            _emit_rows(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
            page_rows = len(rows)
            if page_rows:
                status = f"Exact-search rows {offset + 1}–{offset + page_rows} of {total_count}"
            else:
                status = f"No exact-search rows at offset {offset}; total={total_count}"
            typer.echo(
                f"{status}; scope_id={scope_id}; has_more={str(has_more).lower()}", err=True
            )
    finally:
        instance.close()


@search_app.command("lexical")
def search_lexical(
    project: Path,
    scope_id: str,
    query: str,
    output_format: str = typer.Option("json", "--format", help="json, jsonl, csv, or table."),
    offset: int = typer.Option(0, min=0, max=4_999, help="Zero-based result offset (max 4999)."),
    limit: int = typer.Option(100, min=1, max=100, help="Maximum results in this page."),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """Find scoped FTS5 paragraph matches and return a bounded page."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    if not query.strip() or len(query) > 2_000:
        raise typer.BadParameter("query must contain 1..2000 characters", param_hint="QUERY")
    instance = Project.open(project)
    try:
        payload = instance.search_lexical_page(query, scope_id, offset=offset, limit=limit)
        if output_format == "json":
            _emit(payload)
        else:
            rows = cast(list[dict[str, object]], payload["rows"])
            _emit_rows(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
            typer.echo(
                f"Lexical-search rows {offset + 1}–{offset + len(rows)}; "
                f"has_more={str(payload['has_more']).lower()}; "
                f"candidate_window_full={str(payload['candidate_window_full']).lower()}",
                err=True,
            )
    finally:
        instance.close()


@report_app.command("finding")
def report_finding(
    project: Path,
    finding: str,
    output_format: str = typer.Option("markdown", "--format", help="markdown, html, or json."),
) -> None:
    """Render a finding only from persisted, cited evidence."""
    instance = Project.open(project)
    try:
        if output_format == "markdown":
            _echo_report_document(render_finding_markdown(instance, finding))
        elif output_format == "html":
            _echo_report_document(render_finding_html(instance, finding))
        elif output_format == "json":
            _emit(finding_payload(instance, finding))
        else:
            raise typer.BadParameter(
                "format must be markdown, html, or json", param_hint="--format"
            )
    finally:
        instance.close()


@report_app.command("analysis")
def report_analysis(
    project: Path,
    analysis_run: str,
    output_format: str = typer.Option("markdown", "--format", help="markdown, html, or json."),
) -> None:
    """Render a whole immutable analysis run with a secret-free technical appendix."""
    instance = Project.open(project)
    try:
        if output_format == "markdown":
            _echo_report_document(render_analysis_markdown(instance, analysis_run))
        elif output_format == "html":
            _echo_report_document(render_analysis_html(instance, analysis_run))
        elif output_format == "json":
            _emit(analysis_payload(instance, analysis_run))
        else:
            raise typer.BadParameter(
                "format must be markdown, html, or json", param_hint="--format"
            )
    finally:
        instance.close()


@report_app.command("export")
def report_export(
    project: Path,
    analysis_run: str,
    destination: Path,
    output_format: str = typer.Option(..., "--format", help="markdown, html, json, pdf, or docx."),
) -> None:
    """Write a local evidence-validated analysis report in the requested format."""
    instance = Project.open(project)
    try:
        output = write_analysis_report(instance, analysis_run, destination, output_format)
        _emit({"status": "SUCCESS", "format": output_format, "path": str(output)})
    finally:
        instance.close()


@report_app.command("comparison")
def report_comparison(
    project: Path,
    comparison_run: str,
    output_format: str = typer.Option("markdown", "--format", help="markdown or json."),
    limit: int = typer.Option(25, min=1, max=1_000, help="Maximum ranked rows in a Markdown page."),
    offset: int = typer.Option(0, min=0, help="Ranked result offset for Markdown pages."),
) -> None:
    """Render a persisted bilateral comparison with both populations visible."""
    instance = Project.open(project)
    try:
        if output_format == "markdown":
            _echo_report_document(
                render_comparison_markdown(instance, comparison_run, limit=limit, offset=offset)
            )
        elif output_format == "json":
            _emit(comparison_payload(instance, comparison_run))
        else:
            raise typer.BadParameter("format must be markdown or json", param_hint="--format")
    finally:
        instance.close()


@export_app.command("research-bundle")
def export_research_bundle(
    project: Path,
    analysis_run: str,
    destination: Path,
    include_canonical_text: bool = typer.Option(
        False,
        "--include-canonical-text",
        help=(
            "Embed canonical text for revisions with cited text evidence. Off by default "
            "for privacy; cited excerpts are still present and can be long. The manifest "
            "is limited to 64 MiB."
        ),
    ),
) -> None:
    """Export a manifest; full text is opt-in, but cited excerpts remain included."""
    instance = Project.open(project)
    try:
        output = export_bundle(
            instance,
            analysis_run,
            destination,
            include_canonical_text=include_canonical_text,
        )
        _emit({"status": "SUCCESS", "manifest": str(output)})
    finally:
        instance.close()


@export_app.command("verify-bundle")
def verify_research_bundle(manifest: Path) -> None:
    """Verify bundle structure and its local integrity digest without a project."""
    _emit(verify_bundle(manifest))


@analyze_app.command("textual-descriptors")
def analyze_textual_descriptors(
    project: Path,
    revision: str,
    languages: str = typer.Option("en,es", help="Comma-separated supported marker languages."),
) -> None:
    """Run deterministic modal-marker observations over one or more revisions."""
    revision_ids = [revision_id.strip() for revision_id in revision.split(",")]
    if any(not revision_id for revision_id in revision_ids):
        raise typer.BadParameter(
            "revision IDs must be nonempty and must not be blank", param_hint="REVISION"
        )
    if len(set(revision_ids)) != len(revision_ids):
        raise typer.BadParameter("revision IDs must not be duplicated", param_hint="REVISION")
    selected_languages = tuple(
        language.strip() for language in languages.split(",") if language.strip()
    )
    instance = Project.open(project)
    try:
        _emit(run_textual_descriptors_analysis(instance, revision_ids, selected_languages))
    finally:
        instance.close()


@analyze_app.command("evidence-extraction")
def analyze_evidence_extraction(
    project: Path,
    revisions: str,
    languages: str = typer.Option("en,es", help="Comma-separated cue-inventory languages."),
    max_examples: int = typer.Option(48, min=1, max=256),
) -> None:
    """Create a bounded descriptive candidate inventory for an explicit revision scope."""
    revision_ids = [revision.strip() for revision in revisions.split(",") if revision.strip()]
    selected_languages = tuple(
        language.strip() for language in languages.split(",") if language.strip()
    )
    instance = Project.open(project)
    try:
        _emit(
            run_evidence_extraction_analysis(
                instance,
                revision_ids,
                selected_languages,
                max_examples=max_examples,
            )
        )
    finally:
        instance.close()


@analyze_app.command("batch-textual-descriptors")
def analyze_batch_textual_descriptors(
    project: Path,
    revisions: str,
    languages: str = typer.Option("en,es", help="Comma-separated supported marker languages."),
) -> None:
    """Analyze each selected revision independently; documents are never concatenated."""
    revision_ids = [revision.strip() for revision in revisions.split(",") if revision.strip()]
    if not revision_ids:
        raise typer.BadParameter(
            "revisions must include at least one revision", param_hint="revisions"
        )
    selected_languages = tuple(
        language.strip() for language in languages.split(",") if language.strip()
    )
    instance = Project.open(project)
    try:
        _emit(run_batch_textual_descriptors_analysis(instance, revision_ids, selected_languages))
    finally:
        instance.close()


@analyze_app.command("counterevidence-lexical")
def analyze_counterevidence_lexical(
    project: Path,
    finding: str,
    phrase: str,
    offset: int = typer.Option(0, min=0, help="Zero-based candidate offset."),
    limit: int = typer.Option(100, min=1, max=1_000, help="Maximum candidates in this page."),
    retrieval_run_id: str | None = typer.Option(
        None, "--retrieval-run", help="Continue paging a completed counterevidence scan."
    ),
) -> None:
    """Search the finding's original scope and return a bounded candidate page."""
    instance = Project.open(project)
    try:
        _emit(
            run_counterevidence_review(
                instance,
                finding,
                phrase,
                offset=offset,
                limit=limit,
                retrieval_run_id=retrieval_run_id,
            )
        )
    finally:
        instance.close()


def _validate_parser_candidate_selection(revision_ids: list[str], artifact_ids: list[str]) -> None:
    if not revision_ids:
        raise typer.BadParameter("at least one revision must be selected", param_hint="--revision")
    if any(
        not revision_id.strip() or revision_id != revision_id.strip()
        for revision_id in revision_ids
    ):
        raise typer.BadParameter(
            "revision IDs must be nonempty and must not have surrounding whitespace",
            param_hint="--revision",
        )
    if len(set(revision_ids)) != len(revision_ids):
        raise typer.BadParameter("revision IDs must not be duplicated", param_hint="--revision")
    if not artifact_ids:
        raise typer.BadParameter(
            "at least one token artifact must be selected", param_hint="--artifact"
        )
    if any(
        not artifact_id.strip() or artifact_id != artifact_id.strip()
        for artifact_id in artifact_ids
    ):
        raise typer.BadParameter(
            "artifact IDs must be nonempty and must not have surrounding whitespace",
            param_hint="--artifact",
        )
    if len(set(artifact_ids)) != len(artifact_ids):
        raise typer.BadParameter("artifact IDs must not be duplicated", param_hint="--artifact")


@analyze_app.command("social-actor-candidates")
def analyze_social_actor_candidates(
    project: Path,
    revision_ids: list[str] = typer.Option(
        ...,
        "--revision",
        help="One selected immutable document revision; repeat to include more revisions.",
    ),
    artifact_ids: list[str] = typer.Option(
        ...,
        "--artifact",
        help="One selected parsed-token artifact; repeat to include more artifacts.",
    ),
) -> None:
    """Create evidence-bound parser candidates, not automatic social-actor interpretations."""
    _validate_parser_candidate_selection(revision_ids, artifact_ids)
    instance = Project.open(project)
    try:
        _emit(instance.analyze_social_actor_candidates(revision_ids, artifact_ids))
    finally:
        instance.close()


@analyze_app.command("agency-candidates")
def analyze_agency_candidates(
    project: Path,
    revision_ids: list[str] = typer.Option(
        ...,
        "--revision",
        help="One selected immutable document revision; repeat to include more revisions.",
    ),
    artifact_ids: list[str] = typer.Option(
        ...,
        "--artifact",
        help="One selected parsed-token artifact; repeat to include more artifacts.",
    ),
) -> None:
    """Create dependency-agency candidates without automatically interpreting author intent."""
    _validate_parser_candidate_selection(revision_ids, artifact_ids)
    instance = Project.open(project)
    try:
        _emit(instance.analyze_agency_candidates(revision_ids, artifact_ids))
    finally:
        instance.close()


@analyze_app.command("evidence-review")
def analyze_evidence_review(
    project: Path,
    finding: str,
    preset: str,
    framework: str | None = typer.Option(None, help="Optional framework review ID."),
    reranking_provider: str | None = typer.Option(
        None, help="Optional explicit reviewed reranker provider."
    ),
    reranking_credential: str | None = typer.Option(
        None, help="Named project credential alias for the reranker."
    ),
    reranking_model: str | None = typer.Option(None, help="Explicit reranker model ID."),
    reranking_top_n: int | None = typer.Option(
        None, min=1, help="Optional maximum number of reranked passages passed to review."
    ),
    context_evidence_ids: list[str] | None = typer.Option(
        None,
        "--context-evidence-id",
        help="Persisted EXTERNAL_SOURCE or USER_CONTEXT evidence ID; repeat up to 10 times.",
    ),
) -> None:
    """Run a JSON-validated model review that can cite only persisted evidence IDs."""
    _validate_reranking_cli_selection(
        reranking_provider, reranking_credential, reranking_model, reranking_top_n
    )
    instance = Project.open(project)
    try:
        reranking_gateway = (
            build_reranking_gateway(instance, reranking_provider, reranking_credential)
            if reranking_provider and reranking_credential
            else None
        )
        _emit(
            review_finding_with_model(
                instance,
                finding,
                preset,
                framework=framework,
                reranking_gateway=reranking_gateway,
                reranking_provider_id=reranking_provider,
                reranking_credential_alias=reranking_credential,
                reranking_model=reranking_model,
                reranking_top_n=reranking_top_n,
                context_evidence_ids=tuple(context_evidence_ids or ()),
            ).__dict__
        )
    finally:
        instance.close()


@analyze_app.command("controlled-cda-review")
def analyze_controlled_cda_review(
    project: Path,
    revisions: str = typer.Option(..., help="Comma-separated immutable revision IDs."),
    preset: str = typer.Option(..., help="Explicit saved model preset."),
    routing_profile: str | None = typer.Option(
        None, help="Optional saved profile mapping framework stages to model presets."
    ),
    frameworks: str | None = typer.Option(None, help="Comma-separated framework profile IDs."),
    counterevidence_json: str | None = typer.Option(
        None, help="Optional JSON mapping of phenomenon to lexical review query."
    ),
    reranking_provider: str | None = typer.Option(None, help="Optional reranker provider."),
    reranking_credential: str | None = typer.Option(
        None, help="Named project credential alias for the reranker."
    ),
    reranking_model: str | None = typer.Option(None, help="Explicit reranker model ID."),
    reranking_top_n: int | None = typer.Option(None, min=1),
) -> None:
    """Run descriptive, optional counterevidence, then evidence-bound framework reviews."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    selected_frameworks = (
        tuple(item.strip() for item in frameworks.split(",") if item.strip())
        if frameworks is not None
        else None
    )
    try:
        counterevidence = json.loads(counterevidence_json) if counterevidence_json else None
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("counterevidence_json must be a JSON object") from exc
    if counterevidence is not None and (
        not isinstance(counterevidence, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in counterevidence.items()
        )
    ):
        raise typer.BadParameter("counterevidence_json must be a JSON object")
    _validate_reranking_cli_selection(
        reranking_provider, reranking_credential, reranking_model, reranking_top_n
    )
    instance = Project.open(project)
    try:
        reranking_gateway = (
            build_reranking_gateway(instance, reranking_provider, reranking_credential)
            if reranking_provider and reranking_credential
            else None
        )
        if selected_frameworks is not None:
            result = run_controlled_cda_review(
                instance,
                revision_ids,
                preset,
                frameworks=selected_frameworks,
                counterevidence_queries=counterevidence,
                routing_profile=routing_profile,
                reranking_gateway=reranking_gateway,
                reranking_provider_id=reranking_provider,
                reranking_credential_alias=reranking_credential,
                reranking_model=reranking_model,
                reranking_top_n=reranking_top_n,
            )
        else:
            result = run_controlled_cda_review(
                instance,
                revision_ids,
                preset,
                counterevidence_queries=counterevidence,
                routing_profile=routing_profile,
                reranking_gateway=reranking_gateway,
                reranking_provider_id=reranking_provider,
                reranking_credential_alias=reranking_credential,
                reranking_model=reranking_model,
                reranking_top_n=reranking_top_n,
            )
        _emit(result)
    finally:
        instance.close()


@analyze_app.command("resume-controlled-cda-review")
def analyze_resume_controlled_cda_review(project: Path, analysis_run: str) -> None:
    """Resume a failed controlled CDA run without replaying completed stages."""
    instance = Project.open(project)
    try:
        _emit(resume_controlled_cda_review(instance, analysis_run))
    finally:
        instance.close()


@analyze_app.command("batch-controlled-cda-review")
def analyze_batch_controlled_cda_review(
    project: Path,
    revisions: str = typer.Option(..., help="Comma-separated immutable revision IDs."),
    preset: str = typer.Option(..., help="Explicit saved model preset."),
    routing_profile: str | None = typer.Option(
        None, help="Optional saved per-framework routing profile."
    ),
    frameworks: str | None = typer.Option(None, help="Comma-separated framework profile IDs."),
    reranking_provider: str | None = typer.Option(None, help="Optional reranker provider."),
    reranking_credential: str | None = typer.Option(
        None, help="Named project credential alias for the reranker."
    ),
    reranking_model: str | None = typer.Option(None, help="Explicit reranker model ID."),
    reranking_top_n: int | None = typer.Option(None, min=1),
) -> None:
    """Create one controlled CDA run per document; documents are never concatenated."""
    revision_ids = [item.strip() for item in revisions.split(",") if item.strip()]
    selected_frameworks = (
        tuple(item.strip() for item in frameworks.split(",") if item.strip())
        if frameworks is not None
        else DEFAULT_CDA_FRAMEWORKS
    )
    _validate_reranking_cli_selection(
        reranking_provider, reranking_credential, reranking_model, reranking_top_n
    )
    instance = Project.open(project)
    try:
        reranking_gateway = (
            build_reranking_gateway(instance, reranking_provider, reranking_credential)
            if reranking_provider and reranking_credential
            else None
        )
        _emit(
            run_batch_controlled_cda_review(
                instance,
                revision_ids,
                preset,
                frameworks=selected_frameworks,
                routing_profile=routing_profile,
                reranking_gateway=reranking_gateway,
                reranking_provider_id=reranking_provider,
                reranking_credential_alias=reranking_credential,
                reranking_model=reranking_model,
                reranking_top_n=reranking_top_n,
            )
        )
    finally:
        instance.close()


@provider_app.command(
    "remove-key",
    context_settings={"allow_extra_args": True},
)
def provider_remove_key(
    ctx: typer.Context,
    provider: str,
    alias: str,
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip confirmation and delete this provider/alias keyring entry.",
    ),
) -> None:
    """Remove one derived provider/alias entry from the OS keyring."""
    if ctx.args:
        raise typer.BadParameter("unexpected argument; identify the key only by provider and alias")
    reference = keyring_reference_for_credential(provider, alias)
    keyring = KeyringSecretStore()
    if not keyring.contains(reference):
        _emit(
            {
                "provider_id": provider,
                "credential_alias": alias,
                "removed": False,
                "status": "not_found",
            }
        )
        return

    if not yes and not typer.confirm(
        f"Remove keyring entry for provider {provider!r}, alias {alias!r}?", default=False
    ):
        _emit(
            {
                "provider_id": provider,
                "credential_alias": alias,
                "removed": False,
                "status": "cancelled",
            }
        )
        return

    removed = keyring.remove_if_present(reference)
    _emit(
        {
            "provider_id": provider,
            "credential_alias": alias,
            "removed": removed,
            "status": "removed" if removed else "not_found",
        }
    )


@provider_app.command(
    "store-key",
    context_settings={"allow_extra_args": True},
)
def provider_store_key(ctx: typer.Context, provider: str, alias: str) -> None:
    """Interactively store a provider secret in the OS keyring, without project files."""
    if ctx.args:
        raise typer.BadParameter("unexpected argument; enter secrets only at the hidden prompt")
    reference = keyring_reference_for_credential(provider, alias)
    keyring = KeyringSecretStore()
    # Check before prompting so an existing credential is never needlessly exposed at the
    # terminal. set_if_absent checks again immediately before writing. The platform keyring
    # API has no portable atomic create-if-absent operation; concurrent processes targeting
    # the same alias must be avoided.
    keyring.ensure_absent(reference)
    secret = typer.prompt("Provider secret", hide_input=True)
    keyring.set_if_absent(reference, secret)
    _emit(
        {
            "provider_id": provider,
            "credential_alias": alias,
            "secret_reference": reference,
            "stored": True,
        }
    )


@provider_app.command(
    "add-credential",
    context_settings={"allow_extra_args": True},
)
def provider_add_credential(
    ctx: typer.Context,
    project: Path,
    provider: str,
    alias: str,
    secret_reference: str | None = typer.Option(
        None,
        "--secret-reference",
        "--secret-env",
        help=(
            "Secret reference: environment-variable name or keyring:service:account. "
            "Never pass the secret value."
        ),
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Use no authentication; allowed only for an explicit loopback OpenAI-compatible endpoint.",
    ),
    endpoint: str | None = typer.Option(None, help="Optional explicit provider endpoint."),
) -> None:
    """Store a named credential reference without accepting a secret on the command line."""
    if ctx.args:
        raise typer.BadParameter(
            "unexpected positional argument; credentials must be supplied as a reference option"
        )
    if no_auth == (secret_reference is not None):
        raise typer.BadParameter("choose exactly one of --secret-reference or --no-auth")
    _validate_credential_endpoint(endpoint)
    instance = Project.open(project)
    try:
        credential_id = instance.save_credential_reference(
            CredentialReference(
                provider,
                alias,
                NO_AUTH_REFERENCE if no_auth else str(secret_reference),
                endpoint,
            )
        )
        _emit({"credential_id": credential_id, "provider": provider, "alias": alias})
    finally:
        instance.close()


@provider_app.command("list")
def provider_list(
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, jsonl, or csv."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List reviewed provider transports; this never reads credentials or calls a network."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    rows = [
        {
            "provider_id": item.provider_id,
            "transport": item.transport,
            "endpoint_policy": item.endpoint_policy,
            "generation": item.generation,
            "embeddings": item.embeddings,
            "reranking": item.reranking,
        }
        for item in supported_providers()
    ]
    _emit_records(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)


@provider_app.command("list-credentials")
def provider_list_credentials(project: Path) -> None:
    """List credential references without echoing provider endpoint values."""
    instance = Project.open(project)
    try:
        _emit(
            [
                {
                    "id": row["id"],
                    "provider_id": row["provider_id"],
                    "credential_alias": row["credential_alias"],
                    "secret_reference": row["secret_reference"],
                    "endpoint_configured": row["endpoint"] is not None,
                    "created_at": row["created_at"],
                }
                for row in instance.list_credential_references()
            ]
        )
    finally:
        instance.close()


@model_app.command("save-preset")
def model_save_preset(
    project: Path,
    name: str,
    provider: str,
    credential_alias: str,
    model: str,
    reasoning: str | None = typer.Option(None),
    reasoning_mode: str | None = typer.Option(None, "--reasoning-mode"),
    reasoning_budget_tokens: int | None = typer.Option(None, "--reasoning-budget-tokens"),
    parameters_json: str = typer.Option("{}", "--parameters-json"),
) -> None:
    """Save explicit provider/model selection metadata for a future analysis run."""
    instance = Project.open(project)
    try:
        try:
            parameters = json.loads(parameters_json)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter("must be valid JSON") from exc
        if not isinstance(parameters, dict):
            raise typer.BadParameter("must be a JSON object")
        preset_id = instance.save_model_preset(
            ModelPreset(
                name,
                provider,
                credential_alias,
                model,
                reasoning,
                parameters,
                reasoning_mode,
                reasoning_budget_tokens,
            )
        )
        _emit({"preset_id": preset_id, "name": name})
    finally:
        instance.close()


@model_app.command("list-presets")
def model_list_presets(
    project: Path,
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, jsonl, or csv."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List reproducible model-selection metadata without loading secrets."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        records = [dict(row) for row in instance.list_model_presets()]
        if output_format == "table":
            table_fields = (
                "name",
                "provider_id",
                "credential_alias",
                "model",
                "reasoning",
                "reasoning_mode",
            )
            rows = [
                {field: record.get(field) or "—" for field in table_fields} for record in records
            ]
        else:
            rows = records
        _emit_records(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
    finally:
        instance.close()


@model_price_app.command("set")
def model_price_set(
    project: Path,
    provider: str,
    model: str,
    input_per_million_tokens: str = typer.Option(
        ..., "--input-per-million-tokens", help="Input rate per million tokens."
    ),
    output_per_million_tokens: str = typer.Option(
        ..., "--output-per-million-tokens", help="Output rate per million tokens."
    ),
    currency: str = typer.Option(..., "--currency", help="Three-letter currency code."),
    source_reference: str = typer.Option(
        ..., "--source-reference", help="User-supplied source or reference for these rates."
    ),
    rate_version: str = typer.Option(..., "--rate-version", help="Immutable rate version."),
    rate_date: str = typer.Option(..., "--rate-date", help="Rate date in YYYY-MM-DD form."),
    reasoning_per_million_tokens: str | None = typer.Option(
        None,
        "--reasoning-per-million-tokens",
        help="Optional separate additional reasoning-token rate per million tokens.",
    ),
) -> None:
    """Record explicit prices only; this command never looks up provider prices."""
    if _model_price_source_contains_secret(source_reference):
        raise typer.BadParameter(
            "must not contain credentials or secret-like values", param_hint="--source-reference"
        )
    try:
        price = ModelPrice(
            provider_id=provider,
            model=model,
            input_per_million_tokens=input_per_million_tokens,
            output_per_million_tokens=output_per_million_tokens,
            reasoning_per_million_tokens=reasoning_per_million_tokens,
            currency=currency,
            source_reference=source_reference,
            rate_version=rate_version,
            rate_date=rate_date,
        )
    except ProjectError as exc:
        raise typer.BadParameter(_safe_project_error_message(exc)) from None

    instance = Project.open(project)
    try:
        price_id = instance.save_model_price(price)
        _emit(
            {
                "price_id": price_id,
                "provider_id": price.provider_id,
                "model": price.model,
                "rate_version": price.rate_version,
            }
        )
    finally:
        instance.close()


@model_price_app.command("list")
def model_price_list(
    project: Path,
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, jsonl, or csv."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List explicit, versioned price records without provider or credential access."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        query = (
            "SELECT id,provider_id,model,input_per_million_tokens,output_per_million_tokens,"
            "reasoning_per_million_tokens,currency,source_reference,rate_version,rate_date,created_at "
            "FROM model_prices WHERE 1=1"
        )
        parameters: list[str] = []
        if provider is not None:
            query += " AND provider_id=?"
            parameters.append(provider)
        if model is not None:
            query += " AND model=?"
            parameters.append(model)
        query += " ORDER BY provider_id,model,rate_date DESC,created_at DESC,id DESC"
        records = [dict(row) for row in instance.db.execute(query, parameters).fetchall()]
        for record in records:
            record["source_reference"] = _safe_price_source_for_cli(record["source_reference"])
        if output_format == "table":
            table_fields = (
                "provider_id",
                "model",
                "input_per_million_tokens",
                "output_per_million_tokens",
                "reasoning_per_million_tokens",
                "currency",
                "rate_version",
                "rate_date",
                "source_reference",
            )
            rows = [
                {
                    field: record.get(field) if record.get(field) is not None else "—"
                    for field in table_fields
                }
                for record in records
            ]
        else:
            rows = records
        _emit_records(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
    finally:
        instance.close()


@model_app.command("costs")
def model_costs(
    project: Path,
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, jsonl, or csv."
    ),
    offset: int = typer.Option(0, "--offset", min=0, help="Zero-based page offset."),
    limit: int = typer.Option(
        100, "--limit", min=1, max=1_000, help="Maximum rows in this page (1..1000)."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List a bounded page of persisted costs; unestimated records stay null."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        page = instance.model_costs_page(offset=offset, limit=limit)
        records = cast(list[dict[str, object]], page["rows"])
        table_rows = records
        if output_format in {"table", "csv"}:
            table_rows = [
                {
                    **record,
                    "model_price_snapshot": (
                        json.dumps(
                            record["model_price_snapshot"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if record["model_price_snapshot"] is not None
                        else "—"
                    ),
                }
                for record in records
            ]
        if output_format == "json":
            _emit(page)
        elif output_format == "jsonl":
            # The leading metadata line is separate so execution rows retain
            # their established field names and shape.
            _emit({"_page": {key: value for key, value in page.items() if key != "rows"}})
            _emit_rows(records, output_format)
        else:
            _emit_records(table_rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
    finally:
        instance.close()


@model_app.command("set-capabilities")
def model_set_capabilities(
    project: Path,
    provider: str,
    model: str,
    capabilities_json: str = typer.Option(
        ..., "--capabilities-json", help="JSON object declaring this exact model's capabilities."
    ),
    assurance: str = typer.Option("UNASSESSED"),
    source_reference: str | None = typer.Option(None, "--source-reference"),
) -> None:
    """Save a per-model capability declaration; it can only narrow adapter support."""
    try:
        payload = json.loads(capabilities_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("must be valid JSON", param_hint="--capabilities-json") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("must be a JSON object", param_hint="--capabilities-json")
    allowed = {
        "text",
        "system_prompt",
        "vision",
        "structured_output",
        "streaming",
        "embeddings",
        "reasoning_efforts",
        "reasoning_modes",
        "generation_parameters",
    }
    if set(payload) - allowed:
        raise typer.BadParameter(
            "contains unknown capability fields", param_hint="--capabilities-json"
        )
    for name in ("reasoning_efforts", "reasoning_modes", "generation_parameters"):
        if name in payload:
            if not isinstance(payload[name], list) or not all(
                isinstance(item, str) and item.strip() for item in payload[name]
            ):
                raise typer.BadParameter(
                    f"{name} must be a JSON array of non-empty strings",
                    param_hint="--capabilities-json",
                )
            payload[name] = tuple(payload[name])
    for name in (
        "text",
        "system_prompt",
        "vision",
        "structured_output",
        "streaming",
        "embeddings",
    ):
        if name in payload and not isinstance(payload[name], bool):
            raise typer.BadParameter(f"{name} must be boolean", param_hint="--capabilities-json")
    profile = ModelCapabilityProfile(
        provider,
        model,
        ModelCapabilities(**payload),
        assurance,
        source_reference,
    )
    instance = Project.open(project)
    try:
        profile_id = instance.save_model_capability_profile(profile)
        _emit({"capability_profile_id": profile_id, "provider": provider, "model": model})
    finally:
        instance.close()


@model_app.command("list-capabilities")
def model_list_capabilities(
    project: Path,
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, jsonl, or csv."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List exact-model capability declarations stored in the project."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        records = [dict(item) for item in instance.list_model_capability_profiles()]
        if output_format == "table":
            rows: list[dict[str, object]] = []
            for record in records:
                profile = instance.model_capability_profile(
                    str(record["provider_id"]), str(record["model"])
                )
                if profile is None:
                    raise ProjectError("model capability profile disappeared during listing")
                capabilities = profile.capabilities
                summary = {
                    "text": capabilities.text,
                    "system_prompt": capabilities.system_prompt,
                    "vision": capabilities.vision,
                    "structured_output": capabilities.structured_output,
                    "streaming": capabilities.streaming,
                    "embeddings": capabilities.embeddings,
                    "reasoning_efforts": list(capabilities.reasoning_efforts),
                    "reasoning_modes": list(capabilities.reasoning_modes),
                    "generation_parameters": list(capabilities.generation_parameters),
                }
                rows.append(
                    {
                        "provider_id": profile.provider_id,
                        "model": profile.model,
                        "assurance": profile.assurance,
                        "capabilities": json.dumps(summary, sort_keys=True, separators=(",", ":")),
                    }
                )
        else:
            rows = records
        _emit_records(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
    finally:
        instance.close()


@model_app.command("save-routing")
def model_save_routing(
    project: Path,
    name: str,
    routes_json: str = typer.Option(
        ..., help="JSON object mapping framework ID to saved preset name."
    ),
    fallback_preset: str | None = typer.Option(
        None, help="Explicit fallback for unmapped framework IDs."
    ),
) -> None:
    """Save explicit multi-model routing metadata without accepting credentials or secrets."""
    try:
        routes = json.loads(routes_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            "routes_json must be a JSON object", param_hint="--routes-json"
        ) from exc
    if not isinstance(routes, dict) or not all(
        isinstance(stage, str) and isinstance(preset, str) for stage, preset in routes.items()
    ):
        raise typer.BadParameter(
            "routes_json must map strings to strings", param_hint="--routes-json"
        )
    instance = Project.open(project)
    try:
        routing_id = instance.save_model_routing_profile(name, routes, fallback_preset)
        _emit({"routing_profile_id": routing_id, "name": name})
    finally:
        instance.close()


@model_app.command("list-routing")
def model_list_routing(
    project: Path,
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, jsonl, or csv."
    ),
    spreadsheet_safe_csv: bool = _spreadsheet_safe_csv_option(),
) -> None:
    """List saved routing metadata without loading credentials or secrets."""
    _validate_spreadsheet_safe_csv(output_format, spreadsheet_safe_csv)
    instance = Project.open(project)
    try:
        records = [dict(item) for item in instance.list_model_routing_profiles()]
        if output_format == "table":
            rows = []
            for record in records:
                routes, fallback = instance.model_routing_profile(str(record["name"]))
                rows.append(
                    {
                        "name": record["name"],
                        "routes": ", ".join(
                            f"{stage} → {preset}" for stage, preset in sorted(routes.items())
                        ),
                        "fallback_preset": fallback or "—",
                    }
                )
        else:
            rows = records
        _emit_records(rows, output_format, spreadsheet_safe_csv=spreadsheet_safe_csv)
    finally:
        instance.close()


@model_app.command("generate")
def model_generate(
    project: Path,
    preset: str,
    prompt_file: Path = typer.Option(
        ..., help="UTF-8 prompt file; avoids putting research text in shell history."
    ),
    image: list[str] = typer.Option(
        [],
        "--image",
        help=(
            "One local PNG, JPEG, or WebP image (maximum 8 MiB) for buffered Gemini, "
            "OpenAI Chat Completions, or Anthropic Messages when vision is enabled "
            "for the exact model. Compatible endpoints must be confirmed individually."
        ),
    ),
    stream: bool = typer.Option(False, "--stream", help="Stream text as it is generated."),
    output_format: str = typer.Option("json", "--format", help="Output format: json or text."),
) -> None:
    """Call exactly the provider/model selected by a saved preset."""
    if output_format not in {"json", "text"}:
        raise typer.BadParameter("format must be json or text", param_hint="--format")
    if stream and output_format != "text":
        raise typer.BadParameter("--stream requires --format text", param_hint="--format")
    if len(image) > 1:
        raise typer.BadParameter("--image can be supplied only once", param_hint="--image")
    image_path = image[0] if image else None
    if stream and image_path is not None:
        raise typer.BadParameter("--image is not supported with --stream", param_hint="--image")
    image_input = None
    if image_path is not None:
        try:
            image_input = GenerationImage.from_file(image_path)
        except ModelProviderError as exc:
            raise typer.BadParameter(str(exc), param_hint="--image") from None
    instance = Project.open(project)
    try:
        prompt = _read_utf8_input(prompt_file, label="prompt file")
        if stream:
            chunks = execute_model_preset_stream(instance, preset, prompt)
            emitted = False
            try:
                for chunk in chunks:
                    if chunk:
                        click.echo(chunk, nl=False)
                        sys.stdout.flush()
                        emitted = True
                if emitted:
                    click.echo()
            except ProjectError as exc:
                if emitted:
                    click.echo()
                click.echo(f"Error: {_safe_project_error_message(exc)}", err=True)
                raise typer.Exit(1) from None
            finally:
                close = getattr(chunks, "close", None)
                if callable(close):
                    close()
            return

        result = (
            execute_model_preset(instance, preset, prompt)
            if image_input is None
            else execute_model_preset(instance, preset, prompt, image=image_input)
        )
        if output_format == "text":
            typer.echo(result.response.text)
        else:
            _emit(
                {
                    "execution_id": result.execution_id,
                    "provider": result.response.provider,
                    "model": result.response.model,
                    "text": result.response.text,
                }
            )
    finally:
        instance.close()


@model_app.command("preview-request")
def model_preview_request(
    project: Path,
    preset: str,
    prompt_file: Path,
    include_prompt: bool = typer.Option(
        False, help="Include prompt text in local output; it is never persisted."
    ),
) -> None:
    """Inspect destination and payload metadata before any model request is sent."""
    prompt = _read_utf8_input(prompt_file, label="prompt file")
    instance = Project.open(project)
    try:
        payload = asdict(preview_model_preset_request(instance, preset, prompt))
        if include_prompt:
            payload["prompt"] = prompt
        _emit(payload)
    finally:
        instance.close()


@evidence_app.command("add-external-source")
def evidence_add_external_source(
    project: Path,
    source: str,
    retrieved_at: str,
    excerpt_file: str = typer.Argument(
        ..., help="UTF-8 excerpt file, or '-' to read from stdin (no inline excerpt argument)."
    ),
    title: str | None = typer.Option(None, help="Optional source title."),
    url: str | None = typer.Option(
        None,
        help="Optional provenance URL; avoid signed URLs or secret-bearing query strings. Never sent to the model.",
    ),
) -> None:
    """Store a user-supplied external excerpt; this command never fetches URLs."""
    excerpt = _read_context_file(excerpt_file)
    instance = Project.open(project)
    try:
        evidence_id = instance.external_source_evidence(
            source, excerpt, retrieved_at=retrieved_at, title=title, url=url
        )
        _emit({"evidence_id": evidence_id, "kind": "EXTERNAL_SOURCE"})
    finally:
        instance.close()


@evidence_app.command("add-user-context")
def evidence_add_user_context(project: Path, label: str, content_file: str) -> None:
    """Store bounded researcher context from UTF-8 file or stdin; never corpus text."""
    content = _read_context_file(content_file)
    instance = Project.open(project)
    try:
        evidence_id = instance.user_context_evidence(label, content)
        _emit({"evidence_id": evidence_id, "kind": "USER_CONTEXT"})
    finally:
        instance.close()


@finding_app.command("validate")
def finding_validate(project: Path, finding: str) -> None:
    """Validate evidence existence, canonical hashes, and historical scope membership."""
    instance = Project.open(project)
    try:
        result = validate_finding(instance, finding)
        _emit(asdict(result))
        if not result.valid:
            raise typer.Exit(code=2)
    finally:
        instance.close()


@finding_app.command("review")
def finding_review(
    project: Path,
    finding: str,
    status: str,
    note: str | None = typer.Option(
        None, help="Optional human review note; does not alter generated content."
    ),
) -> None:
    """Append a human acceptance, modification, or rejection decision."""
    instance = Project.open(project)
    try:
        review_id = instance.review_finding(finding, status, note)
        _emit({"review_id": review_id, "finding_id": finding, "status": status.upper()})
    finally:
        instance.close()


@finding_app.command("reviews")
def finding_reviews(project: Path, finding: str) -> None:
    """Show append-only human review history for one persisted finding."""
    instance = Project.open(project)
    try:
        _emit([dict(row) for row in instance.finding_reviews(finding)])
    finally:
        instance.close()


@finding_app.command("sources")
def finding_sources(project: Path, finding: str) -> None:
    """Show source finding IDs and relation types for a synthesized finding."""
    instance = Project.open(project)
    try:
        sources = [
            {
                "source_finding_id": str(row["source_finding_id"]),
                "relation": str(row["relation"]),
            }
            for row in instance.finding_sources(finding)
        ]
        _emit({"finding_id": finding, "sources": sources})
    finally:
        instance.close()


@vector_app.command("build-paragraphs")
def vector_build_paragraphs(project: Path, preset: str, revision: str) -> None:
    """Create a local paragraph vector index using exactly one selected embedding preset."""
    instance = Project.open(project)
    try:
        _emit({"artifact_id": build_paragraph_embedding_index(instance, preset, revision)})
    finally:
        instance.close()


@vector_app.command("preview-paragraphs")
def vector_preview_paragraphs(
    project: Path,
    preset: str,
    revision: str,
    offset: int = typer.Option(0, min=0, help="First paragraph index in this preview window."),
    limit: int = typer.Option(
        20, min=1, max=1000, help="Maximum paragraphs to inspect in this preview window."
    ),
    include_text: bool = typer.Option(
        False, help="Show exact provider-prepared paragraph text and request body locally."
    ),
) -> None:
    """Inspect a paginated embedding request without resolving secrets or using the network."""
    instance = Project.open(project)
    try:
        preview = preview_paragraph_embedding_request(
            instance,
            preset,
            revision,
            offset=offset,
            limit=limit,
            include_text=include_text,
        )
        _emit(asdict(preview))
    finally:
        instance.close()


def _vector_search_scope(instance: Project, scope_or_revision_id: str) -> str:
    """Resolve a vector query's explicit scope, preserving revision shorthand."""
    existing_scope = instance.db.execute(
        "SELECT 1 FROM scope_snapshots WHERE id=?", (scope_or_revision_id,)
    ).fetchone()
    if existing_scope is not None:
        # Validate the persisted scope before passing it through to Core.
        instance.scope_revisions(scope_or_revision_id)
        return scope_or_revision_id
    return instance.snapshot([scope_or_revision_id])


def _vector_hit_payload(hit: Any) -> dict[str, object]:
    """Serialize one hit without repeating preset-run metadata on every row."""
    payload = asdict(hit)
    payload.pop("retrieval_run_id", None)
    return payload


@vector_app.command("hybrid-search")
def vector_hybrid_search(
    project: Path,
    scope: str,
    artifacts: str,
    lexical_query: str,
    vector_json: str = typer.Option(
        ..., help="JSON numeric vector; use a model preset flow for embedding generation."
    ),
    limit: int = typer.Option(10),
) -> None:
    """Fuse exact lexical candidates and local vector artifacts in a revision or scope."""
    try:
        raw_vector = json.loads(vector_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("vector_json must be a JSON array of finite numbers") from exc
    if not isinstance(raw_vector, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_vector
    ):
        raise typer.BadParameter("vector_json must be a JSON array of finite numbers")
    try:
        vector = tuple(float(value) for value in raw_vector)
    except OverflowError:
        raise typer.BadParameter("vector_json must be a JSON array of finite numbers") from None
    if not vector or any(not math.isfinite(value) for value in vector):
        raise typer.BadParameter("vector_json must be a JSON array of finite numbers")
    artifact_ids = [item.strip() for item in artifacts.split(",") if item.strip()]
    instance = Project.open(project)
    try:
        scope_id = _vector_search_scope(instance, scope)
        result = hybrid_search_with_provenance(
            instance,
            scope_id,
            artifact_ids,
            lexical_query,
            vector,
            limit=limit,
        )
        _emit(
            {
                "scope_id": result.scope_id,
                "retrieval_run_id": result.retrieval_run_id,
                "hits": [_vector_hit_payload(item) for item in result.hits],
            }
        )
    finally:
        instance.close()


@vector_app.command("search")
def vector_search(
    project: Path,
    preset: str,
    scope: str,
    artifacts: str,
    query_file: Path = typer.Option(
        ..., help="UTF-8 query file; avoids research text in shell history."
    ),
    limit: int = typer.Option(10, min=1),
) -> None:
    """Search selected vector artifacts inside one revision or an explicit scope."""
    query_text = _read_utf8_input(
        query_file, label="query file", max_characters=MAX_PRESET_QUERY_CHARS
    )
    instance = Project.open(project)
    try:
        result = semantic_search_with_preset(
            instance,
            preset,
            _vector_search_scope(instance, scope),
            [artifact.strip() for artifact in artifacts.split(",") if artifact.strip()],
            query_text,
            limit=limit,
        )
        _emit(
            {
                "scope_id": result.scope_id,
                "retrieval_run_id": result.retrieval_run_id,
                "hits": [_vector_hit_payload(hit) for hit in result.hits],
            }
        )
    finally:
        instance.close()


@vector_app.command("hybrid-search-preset")
def vector_hybrid_search_with_preset(
    project: Path,
    preset: str,
    scope: str,
    artifacts: str,
    query_file: Path = typer.Option(
        ..., help="UTF-8 query file; query text may be sent to the preset's embedding provider."
    ),
    limit: int = typer.Option(10, min=1),
    rrf_constant: int = typer.Option(60, min=1),
) -> None:
    """Fuse lexical and semantic results inside a revision or an explicit scope."""
    query_text = _read_utf8_input(
        query_file, label="query file", max_characters=MAX_PRESET_QUERY_CHARS
    )
    instance = Project.open(project)
    try:
        result = hybrid_search_with_preset(
            instance,
            preset,
            _vector_search_scope(instance, scope),
            [artifact.strip() for artifact in artifacts.split(",") if artifact.strip()],
            query_text,
            limit=limit,
            rrf_constant=rrf_constant,
        )
        _emit(
            {
                "scope_id": result.scope_id,
                "retrieval_run_id": result.retrieval_run_id,
                "hits": [_vector_hit_payload(hit) for hit in result.hits],
            }
        )
    finally:
        instance.close()
