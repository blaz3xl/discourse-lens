"""Measure append, delete, rebuild, and copied round-trip mechanics only."""

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
    if path.is_file():
        return path.stat().st_size
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


def _labels(count: int, prefix: str) -> list[str]:
    return [f"{prefix}-{index:08d}" for index in range(count)]


def _sqlite_vec(root: Path, vectors: np.ndarray, appended: np.ndarray) -> dict[str, object]:
    import sqlite_vec

    base_count, dimensions = vectors.shape
    appended_count = len(appended)
    base_ids = _labels(base_count, "base")
    appended_ids = _labels(appended_count, "append")
    query = appended[0]
    path = root / "sqlite-vec.sqlite"
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute(
        f"CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,embedding float[{dimensions}])"
    )
    db.executemany(
        "INSERT INTO vectors VALUES(?,?)",
        [(identifier, sqlite_vec.serialize_float32(vector)) for identifier, vector in zip(base_ids, vectors, strict=True)],
    )
    db.commit()

    started = time.perf_counter()
    db.executemany(
        "INSERT INTO vectors VALUES(?,?)",
        [
            (identifier, sqlite_vec.serialize_float32(vector))
            for identifier, vector in zip(appended_ids, appended, strict=True)
        ],
    )
    db.commit()
    append_seconds = _seconds(started)
    appended_rows = db.execute(
        "SELECT id FROM vectors WHERE embedding MATCH ? AND k=1",
        (sqlite_vec.serialize_float32(query),),
    ).fetchall()
    appended_hit = str(appended_rows[0][0]) if appended_rows else None

    started = time.perf_counter()
    db.executemany("DELETE FROM vectors WHERE id=?", [(identifier,) for identifier in appended_ids])
    db.commit()
    delete_seconds = _seconds(started)
    after_delete_rows = db.execute(
        "SELECT id FROM vectors WHERE embedding MATCH ? AND k=10",
        (sqlite_vec.serialize_float32(query),),
    ).fetchall()
    after_delete_ids = [str(row[0]) for row in after_delete_rows]
    after_delete_base_rows = db.execute(
        "SELECT id FROM vectors WHERE embedding MATCH ? AND k=10",
        (sqlite_vec.serialize_float32(vectors[0]),),
    ).fetchall()
    after_delete_base_ids = [str(row[0]) for row in after_delete_base_rows]
    rows_after_delete = int(db.execute("SELECT count(*) FROM vectors").fetchone()[0])
    db.close()

    rebuilt_path = root / "sqlite-vec-rebuilt.sqlite"
    started = time.perf_counter()
    rebuilt = sqlite3.connect(rebuilt_path)
    rebuilt.enable_load_extension(True)
    sqlite_vec.load(rebuilt)
    rebuilt.enable_load_extension(False)
    rebuilt.execute(
        f"CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,embedding float[{dimensions}])"
    )
    rebuilt.executemany(
        "INSERT INTO vectors VALUES(?,?)",
        [(identifier, sqlite_vec.serialize_float32(vector)) for identifier, vector in zip(base_ids, vectors, strict=True)],
    )
    rebuilt.commit()
    rebuild_seconds = _seconds(started)
    before_copy_rows = rebuilt.execute(
        "SELECT id FROM vectors WHERE embedding MATCH ? AND k=10",
        (sqlite_vec.serialize_float32(vectors[0]),),
    ).fetchall()
    before_copy_ids = [str(row[0]) for row in before_copy_rows]
    rebuilt.close()

    copied_path = root / "sqlite-vec-copy.sqlite"
    started = time.perf_counter()
    shutil.copy2(rebuilt_path, copied_path)
    copy_seconds = _seconds(started)
    started = time.perf_counter()
    reopened = sqlite3.connect(copied_path)
    reopened.enable_load_extension(True)
    sqlite_vec.load(reopened)
    reopened.enable_load_extension(False)
    reopen_seconds = _seconds(started)
    started = time.perf_counter()
    reopened_rows = reopened.execute(
        "SELECT id FROM vectors WHERE embedding MATCH ? AND k=10",
        (sqlite_vec.serialize_float32(vectors[0]),),
    ).fetchall()
    reopen_query_seconds = _seconds(started)
    copied_ids = [str(row[0]) for row in reopened_rows]
    copied_count = int(reopened.execute("SELECT count(*) FROM vectors").fetchone()[0])
    reopened.close()

    checks = {
        "append_query_finds_new_vector": appended_hit == appended_ids[0],
        "delete_removes_appended_rows": rows_after_delete == base_count
        and not any(identifier.startswith("append-") for identifier in after_delete_ids),
        "rebuild_preserves_base_top10": before_copy_ids == after_delete_base_ids,
        "copied_reopen_preserves_top10": before_copy_ids == copied_ids and copied_count == base_count,
    }
    checks["rebuild_keeps_first_base_hit"] = bool(before_copy_ids) and before_copy_ids[0] == base_ids[0]
    return {
        "append_seconds": append_seconds,
        "append_count": appended_count,
        "append_query_top1": appended_hit,
        "delete_seconds": delete_seconds,
        "rows_after_delete": rows_after_delete,
        "deleted_ids_absent_from_query": not any(identifier.startswith("append-") for identifier in after_delete_ids),
        "rebuild_seconds": rebuild_seconds,
        "rebuild_query_top10": before_copy_ids,
        "rebuilt_bytes": _size(rebuilt_path),
        "copy_seconds": copy_seconds,
        "copied_bytes": _size(copied_path),
        "reopen_seconds": reopen_seconds,
        "reopen_query_seconds": reopen_query_seconds,
        "reopened_rows": copied_count,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _faiss_label(value: int, base_count: int) -> str:
    if value < 0:
        return ""
    if value < base_count:
        return f"base-{value:08d}"
    return f"append-{value - base_count:08d}"


def _faiss(root: Path, vectors: np.ndarray, appended: np.ndarray) -> dict[str, object]:
    import faiss

    base_count, dimensions = vectors.shape
    appended_count = len(appended)
    query = appended[0]
    base_ids = np.arange(base_count, dtype=np.int64)
    appended_ids = np.arange(base_count, base_count + appended_count, dtype=np.int64)
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(dimensions))
    index.add_with_ids(vectors, base_ids)

    started = time.perf_counter()
    index.add_with_ids(appended, appended_ids)
    append_seconds = _seconds(started)
    _, appended_results = index.search(query.reshape(1, -1), 1)
    appended_hit = _faiss_label(int(appended_results[0][0]), base_count)

    started = time.perf_counter()
    removed = int(index.remove_ids(appended_ids))
    delete_seconds = _seconds(started)
    _, after_delete_results = index.search(query.reshape(1, -1), 10)
    after_delete_ids = [_faiss_label(int(value), base_count) for value in after_delete_results[0] if value >= 0]
    rows_after_delete = int(index.ntotal)

    started = time.perf_counter()
    rebuilt = faiss.IndexIDMap2(faiss.IndexFlatL2(dimensions))
    rebuilt.add_with_ids(vectors, base_ids)
    rebuild_seconds = _seconds(started)
    _, before_copy_results = rebuilt.search(vectors[0].reshape(1, -1), 10)
    before_copy_ids = [_faiss_label(int(value), base_count) for value in before_copy_results[0] if value >= 0]

    rebuilt_path = root / "faiss-rebuilt.index"
    faiss.write_index(rebuilt, str(rebuilt_path))
    copied_path = root / "faiss-copy.index"
    started = time.perf_counter()
    shutil.copy2(rebuilt_path, copied_path)
    copy_seconds = _seconds(started)
    started = time.perf_counter()
    reopened = faiss.read_index(str(copied_path))
    reopen_seconds = _seconds(started)
    started = time.perf_counter()
    _, reopened_results = reopened.search(vectors[0].reshape(1, -1), 10)
    reopen_query_seconds = _seconds(started)
    copied_ids = [_faiss_label(int(value), base_count) for value in reopened_results[0] if value >= 0]

    checks = {
        "append_query_finds_new_vector": appended_hit == "append-00000000",
        "delete_removes_appended_rows": removed == appended_count
        and rows_after_delete == base_count
        and not any(identifier.startswith("append-") for identifier in after_delete_ids),
        "rebuild_preserves_base_top10": before_copy_ids
        and before_copy_ids[0] == "base-00000000",
        "copied_reopen_preserves_top10": before_copy_ids == copied_ids
        and int(reopened.ntotal) == base_count,
    }
    return {
        "index_type": "IndexIDMap2(IndexFlatL2)",
        "append_seconds": append_seconds,
        "append_count": appended_count,
        "append_query_top1": appended_hit,
        "delete_seconds": delete_seconds,
        "removed_count": removed,
        "rows_after_delete": rows_after_delete,
        "deleted_ids_absent_from_query": not any(identifier.startswith("append-") for identifier in after_delete_ids),
        "rebuild_seconds": rebuild_seconds,
        "rebuild_query_top10": before_copy_ids,
        "rebuilt_bytes": _size(rebuilt_path),
        "copy_seconds": copy_seconds,
        "copied_bytes": _size(copied_path),
        "reopen_seconds": reopen_seconds,
        "reopen_query_seconds": reopen_query_seconds,
        "reopened_rows": int(reopened.ntotal),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _lance_rows(vectors: np.ndarray, ids: list[str], batch: str):
    import pyarrow as pa

    dimensions = vectors.shape[1]
    embedding = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), dimensions
    )
    return pa.table(
        {
            "chunk_id": pa.array(ids),
            "revision_id": pa.array(["A"] * len(ids)),
            "batch": pa.array([batch] * len(ids)),
            "embedding": embedding,
        }
    )


def _lancedb(root: Path, vectors: np.ndarray, appended: np.ndarray) -> dict[str, object]:
    import lancedb

    base_count = len(vectors)
    appended_count = len(appended)
    base_ids = _labels(base_count, "base")
    appended_ids = _labels(appended_count, "append")
    query = appended[0]
    path = root / "lancedb"
    database = lancedb.connect(str(path))
    table = database.create_table("chunks", data=_lance_rows(vectors, base_ids, "base"))

    started = time.perf_counter()
    table.add(_lance_rows(appended, appended_ids, "append"))
    append_seconds = _seconds(started)
    appended_rows = table.search(query.tolist()).limit(1).to_list()
    appended_hit = str(appended_rows[0]["chunk_id"]) if appended_rows else None

    started = time.perf_counter()
    table.delete("batch = 'append'")
    delete_seconds = _seconds(started)
    after_delete_rows = table.search(query.tolist()).limit(10).to_list()
    after_delete_ids = [str(row["chunk_id"]) for row in after_delete_rows]
    rows_after_delete = int(table.count_rows())

    rebuilt_path = root / "lancedb-rebuilt"
    rebuilt_database = lancedb.connect(str(rebuilt_path))
    started = time.perf_counter()
    rebuilt_table = rebuilt_database.create_table(
        "chunks", data=_lance_rows(vectors, base_ids, "base")
    )
    rebuild_seconds = _seconds(started)
    before_copy_rows = rebuilt_table.search(vectors[0].tolist()).limit(10).to_list()
    before_copy_ids = [str(row["chunk_id"]) for row in before_copy_rows]
    del rebuilt_table, rebuilt_database, table, database
    gc.collect()

    copied_path = root / "lancedb-copy"
    started = time.perf_counter()
    shutil.copytree(rebuilt_path, copied_path)
    copy_seconds = _seconds(started)
    started = time.perf_counter()
    reopened = lancedb.connect(str(copied_path))
    reopened_table = reopened.open_table("chunks")
    reopen_seconds = _seconds(started)
    started = time.perf_counter()
    reopened_rows = reopened_table.search(vectors[0].tolist()).limit(10).to_list()
    reopen_query_seconds = _seconds(started)
    copied_ids = [str(row["chunk_id"]) for row in reopened_rows]
    copied_count = int(reopened_table.count_rows())
    del reopened_table, reopened
    gc.collect()

    checks = {
        "append_query_finds_new_vector": appended_hit == appended_ids[0],
        "delete_removes_appended_rows": rows_after_delete == base_count
        and not any(identifier.startswith("append-") for identifier in after_delete_ids),
        "rebuild_preserves_base_top10": bool(before_copy_ids)
        and before_copy_ids[0] == base_ids[0],
        "copied_reopen_preserves_top10": before_copy_ids == copied_ids
        and copied_count == base_count,
    }
    return {
        "append_seconds": append_seconds,
        "append_count": appended_count,
        "append_query_top1": appended_hit,
        "delete_seconds": delete_seconds,
        "rows_after_delete": rows_after_delete,
        "deleted_ids_absent_from_query": not any(identifier.startswith("append-") for identifier in after_delete_ids),
        "rebuild_seconds": rebuild_seconds,
        "rebuild_query_top10": before_copy_ids,
        "rebuilt_bytes": _size(rebuilt_path),
        "copy_seconds": copy_seconds,
        "copied_bytes": _size(copied_path),
        "reopen_seconds": reopen_seconds,
        "reopen_query_seconds": reopen_query_seconds,
        "reopened_rows": copied_count,
        "checks": checks,
        "passed": all(checks.values()),
        "rebuild_method": "write a fresh table from live rows; native fragment compaction was not exercised",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--append-count", type=int, help="defaults to 10 percent of --chunks")
    parser.add_argument("--candidate", choices=("sqlite-vec", "lancedb", "faiss"), required=True)
    parser.add_argument("--output", help="write the JSON report here without overwriting")
    arguments = parser.parse_args()
    if arguments.chunks < 1:
        parser.error("--chunks must be positive")
    append_count = arguments.append_count or max(100, arguments.chunks // 10)
    if append_count < 1:
        parser.error("--append-count must be positive")

    rng = np.random.default_rng(20260925)
    vectors = rng.standard_normal((arguments.chunks, arguments.dimensions), dtype=np.float32)
    appended = rng.standard_normal((append_count, arguments.dimensions), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="discourselens-vector-lifecycle-") as temporary_root:
        root = Path(temporary_root)
        candidates = {"sqlite-vec": _sqlite_vec, "lancedb": _lancedb, "faiss": _faiss}
        try:
            candidate_result = candidates[arguments.candidate](root, vectors, appended)
        except Exception as exc:  # noqa: BLE001 - errors are evidence in this spike.
            candidate_result = {"error": f"{type(exc).__name__}: {exc}", "passed": False}
        result: dict[str, object] = {
            "dataset": {
                "chunks": arguments.chunks,
                "dimensions": arguments.dimensions,
                "append_count": append_count,
            },
            "candidate": arguments.candidate,
            "note": "Deterministic random vectors measure index mechanics, not semantic retrieval quality.",
            arguments.candidate: candidate_result,
            "lifecycle_passed": isinstance(candidate_result, dict)
            and candidate_result.get("passed") is True,
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output:
        output_path = _checked_output_path(arguments.output)
        if output_path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if result["lifecycle_passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
