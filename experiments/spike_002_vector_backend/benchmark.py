"""Small local vector-mechanics comparison; random vectors are not semantic quality."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np


def _seconds(start: float) -> float:
    return round(time.perf_counter() - start, 6)


def _size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


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


def _sqlite_vec(root: Path, vectors: np.ndarray, query: np.ndarray) -> dict[str, object]:
    import sqlite_vec

    path = root / "sqlite-vec.sqlite"
    started = time.perf_counter()
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    dimension = vectors.shape[1]
    db.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY,document_id TEXT NOT NULL)")
    db.execute(
        f"CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,embedding float[{dimension}])"
    )
    for index, vector in enumerate(vectors):
        db.execute("INSERT INTO chunks VALUES(?,?)", (f"a-{index}", "A"))
        db.execute("INSERT INTO vectors VALUES(?,?)", (f"a-{index}", sqlite_vec.serialize_float32(vector)))
    db.commit()
    build = _seconds(started)
    started = time.perf_counter()
    # Correct scope is enforced by the project design: one derived index per revision.
    rows = db.execute(
        "SELECT id,distance FROM vectors WHERE embedding MATCH ? AND k=10",
        (sqlite_vec.serialize_float32(query),),
    ).fetchall()
    query_seconds = _seconds(started)
    result_ids = [str(row[0]) for row in rows]
    db.close()
    copied_path = root / "sqlite-vec-copy.sqlite"
    started = time.perf_counter()
    shutil.copy2(path, copied_path)
    copy_seconds = _seconds(started)
    started = time.perf_counter()
    reopened = sqlite3.connect(copied_path)
    reopened.enable_load_extension(True)
    sqlite_vec.load(reopened)
    reopened.enable_load_extension(False)
    reopen_seconds = _seconds(started)
    started = time.perf_counter()
    reopened_rows = reopened.execute(
        "SELECT id,distance FROM vectors WHERE embedding MATCH ? AND k=10",
        (sqlite_vec.serialize_float32(query),),
    ).fetchall()
    reopen_query_seconds = _seconds(started)
    reopened.close()
    reopened_ids = [str(row[0]) for row in reopened_rows]
    return {
        "build_seconds": build,
        "query_seconds": query_seconds,
        "hits": len(rows),
        "bytes": path.stat().st_size,
        "copy_seconds": copy_seconds,
        "copied_bytes": copied_path.stat().st_size,
        "reopen_seconds": reopen_seconds,
        "reopen_query_seconds": reopen_query_seconds,
        "reopen_hits": len(reopened_rows),
        "round_trip_matches": result_ids == reopened_ids,
    }


def _lancedb(root: Path, vectors: np.ndarray, query: np.ndarray) -> dict[str, object]:
    import lancedb
    import pyarrow as pa

    path = root / "lancedb"
    started = time.perf_counter()
    database = lancedb.connect(str(path))
    dimensions = vectors.shape[1]
    embedding = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), dimensions
    )
    table = database.create_table(
        "chunks",
        data=pa.table(
            {
                "chunk_id": pa.array([f"a-{index}" for index in range(len(vectors))]),
                "revision_id": pa.array(["A"] * len(vectors)),
                "embedding": embedding,
            }
        ),
    )
    build = _seconds(started)
    started = time.perf_counter()
    rows = (
        table.search(query.tolist())
        .where("revision_id = 'A'", prefilter=True)
        .limit(10)
        .to_list()
    )
    query_seconds = _seconds(started)
    result_ids = [str(row["chunk_id"]) for row in rows]
    source_bytes = _size(path)
    del table, database
    gc.collect()
    copied_path = root / "lancedb-copy"
    started = time.perf_counter()
    shutil.copytree(path, copied_path)
    copy_seconds = _seconds(started)
    started = time.perf_counter()
    reopened = lancedb.connect(str(copied_path))
    reopened_table = reopened.open_table("chunks")
    reopen_seconds = _seconds(started)
    started = time.perf_counter()
    reopened_rows = (
        reopened_table.search(query.tolist())
        .where("revision_id = 'A'", prefilter=True)
        .limit(10)
        .to_list()
    )
    reopen_query_seconds = _seconds(started)
    reopened_ids = [str(row["chunk_id"]) for row in reopened_rows]
    del reopened_table, reopened
    gc.collect()
    return {
        "build_seconds": build,
        "query_seconds": query_seconds,
        "hits": len(rows),
        "bytes": source_bytes,
        "copy_seconds": copy_seconds,
        "copied_bytes": _size(copied_path),
        "reopen_seconds": reopen_seconds,
        "reopen_query_seconds": reopen_query_seconds,
        "reopen_hits": len(reopened_rows),
        "round_trip_matches": result_ids == reopened_ids,
    }


def _faiss(root: Path, vectors: np.ndarray, query: np.ndarray) -> dict[str, object]:
    import faiss

    path = root / "faiss.index"
    started = time.perf_counter()
    index = faiss.IndexFlatL2(vectors.shape[1])
    # FAISS has no metadata predicate. This is the required pre-filtered, per-scope
    # physical index rather than an unsafe retrieve-everything/filter-after design.
    index.add(vectors)
    faiss.write_index(index, str(path))
    build = _seconds(started)
    started = time.perf_counter()
    _, ids = index.search(query.reshape(1, -1), 10)
    query_seconds = _seconds(started)
    result_ids = [int(value) for value in ids[0] if value >= 0]
    copied_path = root / "faiss-copy.index"
    started = time.perf_counter()
    shutil.copy2(path, copied_path)
    copy_seconds = _seconds(started)
    started = time.perf_counter()
    reopened = faiss.read_index(str(copied_path))
    reopen_seconds = _seconds(started)
    started = time.perf_counter()
    _, reopened_ids_array = reopened.search(query.reshape(1, -1), 10)
    reopen_query_seconds = _seconds(started)
    reopened_ids = [int(value) for value in reopened_ids_array[0] if value >= 0]
    return {
        "build_seconds": build,
        "query_seconds": query_seconds,
        "hits": len(result_ids),
        "bytes": path.stat().st_size,
        "copy_seconds": copy_seconds,
        "copied_bytes": copied_path.stat().st_size,
        "reopen_seconds": reopen_seconds,
        "reopen_query_seconds": reopen_query_seconds,
        "reopen_hits": len(reopened_ids),
        "round_trip_matches": result_ids == reopened_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--candidate", choices=("sqlite-vec", "lancedb", "faiss"))
    parser.add_argument("--output", help="write the JSON report here without overwriting")
    arguments = parser.parse_args()
    rng = np.random.default_rng(20260925)
    vectors = rng.standard_normal((arguments.chunks, arguments.dimensions), dtype=np.float32)
    query = vectors[0].copy()
    with tempfile.TemporaryDirectory(prefix="discourselens-vector-spike-") as temporary_root:
        root = Path(temporary_root)
        results: dict[str, object] = {
            "dataset": {"chunks": arguments.chunks, "dimensions": arguments.dimensions},
            "note": "Random vectors measure mechanics, not semantic retrieval quality.",
        }
        candidates = {"sqlite-vec": _sqlite_vec, "lancedb": _lancedb, "faiss": _faiss}
        if arguments.candidate:
            candidates = {arguments.candidate: candidates[arguments.candidate]}
        for name, function in candidates.items():
            try:
                results[name] = function(root, vectors, query)
            except Exception as exc:  # noqa: BLE001 - any candidate failure is spike data.
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}
        results["persistence_round_trip_passed"] = bool(candidates) and all(
            isinstance(results.get(name), dict)
            and results[name].get("round_trip_matches") is True
            for name in candidates
        )
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if arguments.output:
        output_path = _checked_output_path(arguments.output)
        if output_path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if results["persistence_round_trip_passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
