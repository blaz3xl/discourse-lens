"""Adversarial strict-scope checks; these do not measure semantic quality."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np


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


def _sqlite_vec_check(root: Path, scope_a: np.ndarray, scope_b: np.ndarray, query: np.ndarray) -> dict[str, object]:
    import sqlite_vec

    # Confirm with sqlite-vec itself that the unfiltered query prefers B.
    combined = sqlite3.connect(root / "sqlite-combined.sqlite")
    combined.enable_load_extension(True)
    sqlite_vec.load(combined)
    combined.enable_load_extension(False)
    combined.execute(
        f"CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,embedding float[{scope_a.shape[1]}])"
    )
    combined.executemany(
        "INSERT INTO vectors VALUES(?,?)",
        [
            (f"{scope_id}-{index}", sqlite_vec.serialize_float32(vector))
            for scope_id, vectors in (("A", scope_a), ("B", scope_b))
            for index, vector in enumerate(vectors)
        ],
    )
    unfiltered_rows = combined.execute(
        "SELECT id FROM vectors WHERE embedding MATCH ? AND k=10",
        (sqlite_vec.serialize_float32(query),),
    ).fetchall()
    combined.close()
    unfiltered_scopes = sorted({str(row[0]).split("-", 1)[0] for row in unfiltered_rows})

    for scope_id, vectors in (("A", scope_a), ("B", scope_b)):
        connection = sqlite3.connect(root / f"sqlite-{scope_id}.sqlite")
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        connection.execute(
            f"CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,embedding float[{vectors.shape[1]}])"
        )
        connection.executemany(
            "INSERT INTO vectors VALUES(?,?)",
            [
                (f"{scope_id}-{index}", sqlite_vec.serialize_float32(vector))
                for index, vector in enumerate(vectors)
            ],
        )
        connection.commit()
        connection.close()

    # The scope's approved revision index is selected before the vector query.
    connection = sqlite3.connect(root / "sqlite-A.sqlite")
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    rows = connection.execute(
        "SELECT id FROM vectors WHERE embedding MATCH ? AND k=10",
        (sqlite_vec.serialize_float32(query),),
    ).fetchall()
    connection.close()
    scopes = sorted({str(row[0]).split("-", 1)[0] for row in rows})
    return {
        "strategy": "select the approved per-revision index before search",
        "unfiltered_returned_scopes": unfiltered_scopes,
        "returned_scopes": scopes,
        "passed": bool(unfiltered_rows) and unfiltered_scopes == ["B"] and bool(rows) and scopes == ["A"],
    }


def _faiss_check(scope_a: np.ndarray, scope_b: np.ndarray, query: np.ndarray) -> dict[str, object]:
    import faiss

    combined = faiss.IndexFlatL2(scope_a.shape[1])
    combined.add(np.concatenate((scope_a, scope_b), axis=0))
    _, unfiltered_ids = combined.search(query.reshape(1, -1), 10)
    unfiltered_scopes = sorted(
        {"A" if int(value) < len(scope_a) else "B" for value in unfiltered_ids[0] if value >= 0}
    )

    # FAISS has no metadata predicate; the allowed per-revision index is searched.
    indexes = {}
    for scope_id, vectors in (("A", scope_a), ("B", scope_b)):
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        indexes[scope_id] = index
    _, ids = indexes["A"].search(query.reshape(1, -1), 10)
    returned_scopes = ["A" for value in ids[0] if value >= 0]
    return {
        "strategy": "select the approved per-revision index before search",
        "unfiltered_returned_scopes": unfiltered_scopes,
        "returned_scopes": sorted(set(returned_scopes)),
        "hits": len(returned_scopes),
        "passed": unfiltered_scopes == ["B"] and bool(returned_scopes) and set(returned_scopes) == {"A"},
    }


def _lancedb_check(root: Path, scope_a: np.ndarray, scope_b: np.ndarray, query: np.ndarray) -> dict[str, object]:
    import lancedb
    import pyarrow as pa

    vectors = np.concatenate((scope_a, scope_b), axis=0)
    scope_ids = ["A"] * len(scope_a) + ["B"] * len(scope_b)
    embedding = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), vectors.shape[1]
    )
    database = lancedb.connect(str(root / "lancedb-scope"))
    table = database.create_table(
        "chunks",
        data=pa.table(
            {
                "chunk_id": pa.array(
                    [f"{scope_id}-{index}" for scope_id in ("A", "B") for index in range(10)]
                ),
                "revision_id": pa.array(scope_ids),
                "embedding": embedding,
            }
        ),
    )
    unfiltered_rows = table.search(query.tolist()).limit(10).to_list()
    unfiltered_scopes = sorted({str(row["revision_id"]) for row in unfiltered_rows})
    rows = (
        table.search(query.tolist())
        .where("revision_id = 'A'", prefilter=True)
        .limit(10)
        .to_list()
    )
    scopes = sorted({str(row["revision_id"]) for row in rows})
    return {
        "strategy": "LanceDB metadata predicate with prefilter=True",
        "unfiltered_returned_scopes": unfiltered_scopes,
        "returned_scopes": scopes,
        "passed": bool(unfiltered_rows) and unfiltered_scopes == ["B"] and bool(rows) and scopes == ["A"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("sqlite-vec", "faiss", "lancedb"))
    parser.add_argument("--output", help="write the JSON report here without overwriting")
    arguments = parser.parse_args()

    query = np.zeros(768, dtype=np.float32)
    scope_a = np.zeros((10, 768), dtype=np.float32)
    scope_a[:, 0] = 0.1
    scope_b = np.zeros((10, 768), dtype=np.float32)
    scope_b[:, 0] = 0.01
    nearest_unfiltered_scope = "B" if np.linalg.norm(scope_b[0] - query) < np.linalg.norm(scope_a[0] - query) else "A"
    results: dict[str, object] = {
        "fixture": "Each engine must return B unfiltered, then only A after its scoped retrieval strategy",
        "nearest_unfiltered_scope": nearest_unfiltered_scope,
        "scope_filter_checks": {},
        "note": "This adversarial fixture checks scope mechanics only, not semantic quality.",
    }

    checks: dict[str, object] = results["scope_filter_checks"]  # type: ignore[assignment]
    with tempfile.TemporaryDirectory(prefix="discourselens-vector-scope-") as temporary_root:
        root = Path(temporary_root)
        candidates = {
            "sqlite-vec": lambda: _sqlite_vec_check(root, scope_a, scope_b, query),
            "faiss": lambda: _faiss_check(scope_a, scope_b, query),
            "lancedb": lambda: _lancedb_check(root, scope_a, scope_b, query),
        }
        if arguments.candidate:
            candidates = {arguments.candidate: candidates[arguments.candidate]}
        for name, check in candidates.items():
            try:
                checks[name] = check()
            except Exception as exc:  # noqa: BLE001 - candidate errors are spike results.
                checks[name] = {"error": f"{type(exc).__name__}: {exc}", "passed": False}

    passed = nearest_unfiltered_scope == "B" and all(
        isinstance(check, dict) and check.get("passed") is True
        for check in checks.values()
    )
    results["passed"] = passed
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if arguments.output:
        output_path = _checked_output_path(arguments.output)
        if output_path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
