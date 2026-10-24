"""Compare exact per-revision vector search mechanics across many revisions.

The fixture mirrors ``search_vectors``' retrieval shape: the caller supplies a
revision scope, one physical index is opened per selected revision, each index
returns up to k candidates, and those candidates are globally distance-sorted.
It does not exercise project registries, artifact hash checks, or semantic
embedding quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sqlite3
import tempfile
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np


SEED = 20261001
REVISION_PREFIX = "r"


def _seconds(start: float) -> float:
    return round(time.perf_counter() - start, 6)


def _rss_bytes() -> int | None:
    """Return current process RSS from procfs where available."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _status_memory_bytes(field: str) -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith(f"{field}:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _peak_rss_bytes() -> int | None:
    proc_hwm = _status_memory_bytes("VmHWM")
    if proc_hwm is not None:
        return proc_hwm
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None
    # Linux reports KiB; macOS reports bytes.
    return int(value * 1024 if platform.system() == "Linux" else value)


def _checked_output_path(value: str) -> Path:
    root = Path.cwd().resolve()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit("--output must resolve inside the current working directory") from exc
    return resolved


def _revision_id(index: int) -> str:
    return f"{REVISION_PREFIX}{index:04d}"


def _artifact_paths(root: Path, revisions: int, suffix: str) -> list[Path]:
    return [root / f"{_revision_id(index)}{suffix}" for index in range(revisions)]


def _make_dataset(
    revisions: int, chunks_per_revision: int, dimensions: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    query = rng.standard_normal(dimensions, dtype=np.float32)
    vectors = rng.standard_normal(
        (revisions, chunks_per_revision, dimensions), dtype=np.float32
    )
    if revisions > 1:
        # An exact nearest match in an excluded revision makes the preselection
        # check adversarial without changing the deterministic random fixture.
        vectors[1, 0] = query
    return vectors, query


def _build_sqlite_vec(
    paths: list[Path], vectors: np.ndarray, metric: str
) -> tuple[float, int, list[int]]:
    import sqlite_vec

    revisions, chunks_per_revision, dimensions = vectors.shape
    started = time.perf_counter()
    row_counts: list[int] = []
    for revision_index, path in enumerate(paths):
        connection = sqlite3.connect(path)
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        connection.execute(
            "CREATE TABLE chunks(id TEXT PRIMARY KEY,start_offset INTEGER,end_offset INTEGER)"
        )
        metric_clause = " distance_metric=cosine" if metric == "cosine" else ""
        connection.execute(
            f"CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,"
            f"embedding float[{dimensions}]{metric_clause})"
        )
        for chunk_index, vector in enumerate(vectors[revision_index]):
            chunk_id = f"{_revision_id(revision_index)}-{chunk_index:04d}"
            connection.execute(
                "INSERT INTO chunks VALUES(?,?,?)",
                (chunk_id, chunk_index * 100, chunk_index * 100 + 100),
            )
            connection.execute(
                "INSERT INTO vectors VALUES(?,?)",
                (chunk_id, json.dumps(vector.tolist(), separators=(",", ":"))),
            )
        connection.commit()
        row_counts.append(
            int(connection.execute("SELECT count(*) FROM vectors").fetchone()[0])
        )
        connection.close()
    build_seconds = _seconds(started)
    byte_sizes = [path.stat().st_size for path in paths]
    assert len(row_counts) == revisions and all(
        count == chunks_per_revision for count in row_counts
    )
    return build_seconds, sum(byte_sizes), byte_sizes


def _build_faiss(
    paths: list[Path], vectors: np.ndarray, metric: str
) -> tuple[float, int, list[int]]:
    import faiss

    faiss.omp_set_num_threads(1)
    revisions, chunks_per_revision, dimensions = vectors.shape
    started = time.perf_counter()
    row_counts: list[int] = []
    for revision_index, path in enumerate(paths):
        revision_vectors = np.ascontiguousarray(vectors[revision_index])
        if metric == "cosine":
            norms = np.linalg.norm(revision_vectors, axis=1, keepdims=True)
            indexed_vectors = np.ascontiguousarray(revision_vectors / norms)
            flat_index = faiss.IndexFlatIP(dimensions)
        else:
            indexed_vectors = revision_vectors
            flat_index = faiss.IndexFlatL2(dimensions)
        index = faiss.IndexIDMap2(flat_index)
        index.add_with_ids(
            indexed_vectors, np.arange(chunks_per_revision, dtype=np.int64)
        )
        faiss.write_index(index, str(path))
        row_counts.append(int(index.ntotal))
    build_seconds = _seconds(started)
    byte_sizes = [path.stat().st_size for path in paths]
    assert len(row_counts) == revisions and all(
        count == chunks_per_revision for count in row_counts
    )
    return build_seconds, sum(byte_sizes), byte_sizes


def _hash_files(paths: list[Path]) -> tuple[float, list[str]]:
    started = time.perf_counter()
    digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    return _seconds(started), digests


def _search_sqlite_vec(
    paths: list[Path],
    revision_indices: list[int],
    query: np.ndarray,
    limit: int,
) -> dict[str, Any]:
    import sqlite_vec

    query_json = json.dumps(query.tolist(), separators=(",", ":"))
    hits: list[tuple[float, str, str]] = []
    open_seconds = 0.0
    vector_query_seconds = 0.0
    searched_revision_ids: list[str] = []
    end_to_end_started = time.perf_counter()
    for revision_index in revision_indices:
        revision_id = _revision_id(revision_index)
        open_started = time.perf_counter()
        connection = sqlite3.connect(paths[revision_index])
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        open_seconds += time.perf_counter() - open_started
        searched_revision_ids.append(revision_id)
        query_started = time.perf_counter()
        rows = connection.execute(
            "SELECT v.id,v.distance,c.start_offset,c.end_offset "
            "FROM vectors v JOIN chunks c ON c.id=v.id "
            "WHERE v.embedding MATCH ? AND k=?",
            (query_json, limit),
        ).fetchall()
        vector_query_seconds += time.perf_counter() - query_started
        connection.close()
        hits.extend(
            (float(row[1]), revision_id, str(row[0])) for row in rows
        )
    aggregate_started = time.perf_counter()
    ranked = sorted(hits, key=lambda hit: hit[0])[:limit]
    aggregation_seconds = _seconds(aggregate_started)
    end_to_end_seconds = _seconds(end_to_end_started)
    return {
        "searched_revision_ids": searched_revision_ids,
        "opened_revision_count": len(searched_revision_ids),
        "candidate_count": len(hits),
        "result_count": len(ranked),
        "index_open_seconds": round(open_seconds, 6),
        "vector_query_seconds": round(vector_query_seconds, 6),
        "aggregation_seconds": aggregation_seconds,
        "end_to_end_search_seconds": end_to_end_seconds,
        "top_hits": [
            {"revision_id": revision, "chunk_id": chunk_id, "distance": distance}
            for distance, revision, chunk_id in ranked
        ],
    }


def _search_faiss(
    paths: list[Path],
    revision_indices: list[int],
    query: np.ndarray,
    limit: int,
    metric: str,
) -> dict[str, Any]:
    import faiss

    faiss.omp_set_num_threads(1)
    if metric == "cosine":
        raw_query = np.ascontiguousarray(
            query / np.linalg.norm(query), dtype=np.float32
        ).reshape(1, -1)
    else:
        raw_query = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
    hits: list[tuple[float, str, str]] = []
    open_seconds = 0.0
    vector_query_seconds = 0.0
    searched_revision_ids: list[str] = []
    end_to_end_started = time.perf_counter()
    for revision_index in revision_indices:
        revision_id = _revision_id(revision_index)
        open_started = time.perf_counter()
        index = faiss.read_index(str(paths[revision_index]))
        open_seconds += time.perf_counter() - open_started
        searched_revision_ids.append(revision_id)
        query_started = time.perf_counter()
        scores, local_ids = index.search(raw_query, limit)
        vector_query_seconds += time.perf_counter() - query_started
        hits.extend(
            (
                (
                    1.0 - float(score)
                    if metric == "cosine"
                    else math.sqrt(max(float(score), 0.0))
                ),
                revision_id,
                f"{revision_id}-{int(local_id):04d}",
            )
            for score, local_id in zip(scores[0], local_ids[0], strict=True)
            if local_id >= 0
        )
    aggregate_started = time.perf_counter()
    ranked = sorted(hits, key=lambda hit: hit[0])[:limit]
    aggregation_seconds = _seconds(aggregate_started)
    end_to_end_seconds = _seconds(end_to_end_started)
    return {
        "searched_revision_ids": searched_revision_ids,
        "opened_revision_count": len(searched_revision_ids),
        "candidate_count": len(hits),
        "result_count": len(ranked),
        "index_open_seconds": round(open_seconds, 6),
        "vector_query_seconds": round(vector_query_seconds, 6),
        "aggregation_seconds": aggregation_seconds,
        "end_to_end_search_seconds": end_to_end_seconds,
        "top_hits": [
            {"revision_id": revision, "chunk_id": chunk_id, "distance": distance}
            for distance, revision, chunk_id in ranked
        ],
    }


def _passed(
    selected: dict[str, Any], all_scope: dict[str, Any], top_k: int, revisions: int
) -> bool:
    selected_revision_ids = selected["searched_revision_ids"]
    all_scope_revision_ids = all_scope["searched_revision_ids"]
    return (
        len(selected_revision_ids) == 1
        and selected_revision_ids == [_revision_id(0)]
        and all(len(hit["revision_id"]) > 0 and hit["revision_id"] == _revision_id(0)
                for hit in selected["top_hits"])
        and len(all_scope_revision_ids) == revisions
        and all_scope["opened_revision_count"] == revisions
        and len(selected["top_hits"]) == top_k
        and len(all_scope["top_hits"]) == top_k
        and all_scope["top_hits"][0]["revision_id"] == _revision_id(1)
        and all_scope["top_hits"][0]["chunk_id"] == f"{_revision_id(1)}-0000"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("sqlite-vec", "faiss"), required=True)
    parser.add_argument("--revisions", type=int, default=100)
    parser.add_argument("--chunks-per-revision", type=int, default=1_000)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--metric", choices=("l2", "cosine"), default="l2",
        help="l2 reproduces the legacy vec0 default; cosine models new explicit-metric indexes",
    )
    parser.add_argument("--output", help="write JSON without overwriting an existing file")
    arguments = parser.parse_args()
    if min(
        arguments.revisions,
        arguments.chunks_per_revision,
        arguments.dimensions,
        arguments.top_k,
    ) < 1:
        parser.error("revisions, chunks per revision, dimensions, and top-k must be positive")
    if arguments.revisions < 2:
        parser.error("at least two revisions are required for the strict-preselection fixture")

    vectors, query = _make_dataset(
        arguments.revisions, arguments.chunks_per_revision, arguments.dimensions
    )
    data_bytes = int(vectors.nbytes + query.nbytes)
    baseline_rss_bytes = _rss_bytes()
    with tempfile.TemporaryDirectory(prefix="discourselens-vector-multirevision-") as temp:
        root = Path(temp)
        suffix = ".sqlite" if arguments.candidate == "sqlite-vec" else ".faiss"
        paths = _artifact_paths(root, arguments.revisions, suffix)
        if arguments.candidate == "sqlite-vec":
            build_seconds, persisted_bytes, file_sizes = _build_sqlite_vec(
                paths, vectors, arguments.metric
            )
            search = _search_sqlite_vec
        else:
            build_seconds, persisted_bytes, file_sizes = _build_faiss(
                paths, vectors, arguments.metric
            )

            def search(
                indexed_paths: list[Path],
                revision_indices: list[int],
                search_query: np.ndarray,
                top_k: int,
            ) -> dict[str, Any]:
                return _search_faiss(
                    indexed_paths, revision_indices, search_query, top_k, arguments.metric
                )

        artifact_digests_seconds, artifact_digests = _hash_files(paths)
        # Each request is given its already approved immutable revision list;
        # the backend must open and search only the listed physical indexes.
        selection_started = time.perf_counter()
        requested_selected_revisions = [0]
        selected_revision_indices = [
            index for index in requested_selected_revisions if 0 <= index < len(paths)
        ]
        selected_selection_seconds = _seconds(selection_started)
        selected = search(paths, selected_revision_indices, query, arguments.top_k)

        all_selection_started = time.perf_counter()
        requested_all_revisions = list(range(arguments.revisions))
        all_revision_indices = [
            index for index in requested_all_revisions if 0 <= index < len(paths)
        ]
        all_selection_seconds = _seconds(all_selection_started)
        all_scope = search(paths, all_revision_indices, query, arguments.top_k)

        expected_digest_count = arguments.revisions
        actual_digest_count = len(artifact_digests)
        for request, selection_seconds in (
            (selected, selected_selection_seconds),
            (all_scope, all_selection_seconds),
        ):
            request["scope_selection_seconds"] = selection_seconds
            request["scope_preselection_verified"] = (
                request["searched_revision_ids"]
                == [_revision_id(index) for index in (
                    selected_revision_indices if request is selected else all_revision_indices
                )]
            )

        peak_rss_bytes = _peak_rss_bytes()
        result: dict[str, Any] = {
            "candidate": arguments.candidate,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
                "candidate_package": (
                    "sqlite-vec" if arguments.candidate == "sqlite-vec" else "faiss-cpu"
                ),
                "candidate_version": package_version(
                    "sqlite-vec" if arguments.candidate == "sqlite-vec" else "faiss-cpu"
                ),
                "faiss_openmp_threads": 1 if arguments.candidate == "faiss" else None,
            },
            "dataset": {
                "revisions": arguments.revisions,
                "chunks_per_revision": arguments.chunks_per_revision,
                "total_vectors": arguments.revisions * arguments.chunks_per_revision,
                "dimensions": arguments.dimensions,
                "distance_metric": arguments.metric,
                "seed": SEED,
                "query_vector_source": "independent deterministic seeded vector",
                "excluded_revision_adversary": (
                    f"{_revision_id(1)}-0000 is an exact copy of the query vector"
                ),
                "source_vector_matrix_bytes": data_bytes,
            },
            "build": {
                "seconds": build_seconds,
                "persisted_bytes": persisted_bytes,
                "artifact_count": len(paths),
                "artifact_sizes_bytes": file_sizes,
                "all_artifact_hashes_sha256": artifact_digests,
                "hash_all_artifacts_seconds": artifact_digests_seconds,
                "hash_count": actual_digest_count,
                "hash_count_matches_artifacts": actual_digest_count == expected_digest_count,
            },
            "queries": {"selected_scope": selected, "all_scope": all_scope},
            "process_memory": {
                "baseline_rss_bytes": baseline_rss_bytes,
                "end_rss_bytes": _rss_bytes(),
                "peak_rss_bytes": peak_rss_bytes,
                "peak_rss_source": "Linux /proc/self/status VmHWM" if Path("/proc/self/status").exists() else "resource.getrusage(RUSAGE_SELF)",
                "source_vector_matrix_bytes": data_bytes,
                "note": "Process RSS includes Python, candidate libraries, source vectors, and resident index data. VmHWM is used on Linux because ru_maxrss was anomalously identical across candidates in this WSL environment.",
            },
            "notes": [
                "Index construction mirrors one immutable index file per revision; sqlite-vec includes the chunks join table used by search_vectors.",
                "Search opens only indexes named by the preselected revision scope, gets top-k per index, then globally sorts and truncates to top-k.",
                "L2 mode uses the legacy vec0 default and FAISS IndexFlatL2; FAISS squared L2 values are square-rooted here for score parity.",
                "Cosine mode declares distance_metric=cosine in vec0 and uses normalized vectors with FAISS IndexFlatIP; score is 1 minus cosine similarity.",
                "The 100k default run measures legacy L2 behavior; use --metric cosine for a bounded parity check of new explicit-metric indexes.",
                "Artifact hashing is measured separately; project registry validation, hash-cache behavior, artifact publication, and text-offset validation are not included in search timings.",
                "Random vectors measure backend mechanics, not semantic retrieval quality.",
            ],
        }
        result["passed"] = (
            result["build"]["hash_count_matches_artifacts"]
            and selected["scope_preselection_verified"]
            and all_scope["scope_preselection_verified"]
            and _passed(selected, all_scope, arguments.top_k, arguments.revisions)
        )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output:
        output_path = _checked_output_path(arguments.output)
        if output_path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
