"""Disposable storage benchmark for equivalent synthetic NLP token records."""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sqlite3
import statistics
import time
import zlib
from pathlib import Path

VOCAB=["migration","policy","community","border","worker","public","social","evidence","debate","agency"]
POS=["NOUN","VERB","ADJ","ADP","DET"]
REL=["nsubj","obj","amod","obl","root"]

def records(n:int, seed:int=41):
    offset=0
    for i in range(n):
        surface=VOCAB[i%len(VOCAB)]; sentence=i//20; doc=sentence//50
        yield (doc,sentence,i%20,surface,surface,POS[i%5],"Number=Sing",max(0,i%20-1),REL[i%5],offset,offset+len(surface))
        offset += len(surface)+1

def timed(fn, repeats=3):
    samples=[]; value=None
    for _ in range(repeats):
        t=time.perf_counter(); value=fn(); samples.append(time.perf_counter()-t)
    return value, round(statistics.median(samples)*1000,2)

def sqlite_rows(path, n):
    db=sqlite3.connect(path); db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE token(doc INTEGER,sent INTEGER,idx INTEGER,surface TEXT,lemma TEXT,pos TEXT,morph TEXT,head INTEGER,rel TEXT,start INTEGER,end INTEGER)")
    db.execute("CREATE INDEX token_lemma ON token(lemma)"); db.execute("CREATE INDEX token_doc ON token(doc)")
    db.executemany("INSERT INTO token VALUES(?,?,?,?,?,?,?,?,?,?,?)", records(n)); db.commit()
    queries={
      "lemma_frequency": "SELECT count(*) FROM token WHERE lemma='migration'",
      "document_frequency": "SELECT count(DISTINCT doc) FROM token WHERE lemma='migration'",
      "kwic": "SELECT doc,sent,idx FROM token WHERE lemma='migration' LIMIT 50",
      "pos_filter": "SELECT count(*) FROM token WHERE pos='NOUN'",
      "dependency": "SELECT count(*) FROM token WHERE rel='nsubj'",
      "per_document": "SELECT doc,count(*) FROM token GROUP BY doc",
    }
    out={k:timed(lambda q=q:list(db.execute(q)))[1] for k,q in queries.items()}; db.close(); return out

def sqlite_blobs(path,n):
    db=sqlite3.connect(path); db.execute("CREATE TABLE sentence(doc INTEGER,sent INTEGER,payload BLOB)")
    batch=[]
    for row in records(n):
        batch.append(row)
        if len(batch)==20:
            payload = zlib.compress(json.dumps(batch, separators=(",", ":")).encode())
            db.execute(
                "INSERT INTO sentence VALUES(?,?,?)",
                (batch[0][0], batch[0][1], payload),
            )
            batch = []
    if batch:
        payload = zlib.compress(json.dumps(batch, separators=(",", ":")).encode())
        db.execute(
            "INSERT INTO sentence VALUES(?,?,?)",
            (batch[0][0], batch[0][1], payload),
        )
    db.commit()
    def freq():
      return sum(1 for (payload,) in db.execute("SELECT payload FROM sentence") for row in json.loads(zlib.decompress(payload)) if row[4]=="migration")
    out={"lemma_frequency":timed(freq)[1]}; db.close(); return out

def write_sqlite_blobs_vectorized(path, n, batch_size=100_000):
    """Write equivalent sentence blobs from bounded Arrow fixture batches."""
    from vectorized_parquet import batch as token_batch

    if n % 20 or batch_size % 20:
        raise ValueError("token count and batch size must contain whole sentences")

    fields = ("doc", "sent", "idx", "surface", "lemma", "pos", "morph", "head", "rel", "start", "end")
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE sentence(doc INTEGER,sent INTEGER,payload BLOB)")

    started = time.perf_counter()
    for start in range(0, n, batch_size):
        count = min(batch_size, n - start)
        columns = token_batch(start, count).to_pydict()
        sentences = []
        for sentence_start in range(0, count, 20):
            rows = [
                tuple(columns[field][i] for field in fields)
                for i in range(sentence_start, sentence_start + 20)
            ]
            sentences.append(
                (
                    rows[0][0],
                    rows[0][1],
                    zlib.compress(json.dumps(rows, separators=(",", ":")).encode()),
                )
            )
        db.executemany("INSERT INTO sentence VALUES(?,?,?)", sentences)
        db.commit()
        written = start + count
        if written % 1_000_000 == 0 or written == n:
            print(json.dumps({
                "phase": "write_progress",
                "architecture": "sqlite_blobs_vectorized",
                "tokens_written": written,
                "sentence_rows": db.execute("SELECT count(*) FROM sentence").fetchone()[0],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "size_bytes": Path(path).stat().st_size,
            }), flush=True)
    write_ms = round((time.perf_counter() - started) * 1000, 2)
    sentence_rows = db.execute("SELECT count(*) FROM sentence").fetchone()[0]
    quick_check = db.execute("PRAGMA quick_check").fetchone()[0]
    return db, {
        "write_ms": write_ms,
        "size_bytes": Path(path).stat().st_size,
        "sentence_rows": sentence_rows,
        "quick_check": quick_check,
        "peak_rss_kib_after_write": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }

def sqlite_blob_frequency(db):
    token_count = 0
    matches = 0
    for (payload,) in db.execute("SELECT payload FROM sentence"):
        rows = json.loads(zlib.decompress(payload))
        token_count += len(rows)
        matches += sum(1 for row in rows if row[4] == "migration")
    return {"decoded_tokens": token_count, "lemma_frequency": matches}

def parquet_duckdb(out,n):
    try:
      import duckdb
    except ImportError: return {"unavailable":"duckdb not installed"}
    con=duckdb.connect(config={"threads":"2","memory_limit":"512MB"}); con.execute("CREATE TABLE token(doc BIGINT,sent BIGINT,idx BIGINT,surface VARCHAR,lemma VARCHAR,pos VARCHAR,morph VARCHAR,head BIGINT,rel VARCHAR,start BIGINT,end BIGINT)")
    con.executemany("INSERT INTO token VALUES (?,?,?,?,?,?,?,?,?,?,?)", records(n))
    con.execute("COPY token TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(out)])
    con.from_parquet(str(out)).create_view("p")
    queries={"lemma_frequency":"SELECT count(*) FROM p WHERE lemma='migration'","document_frequency":"SELECT count(DISTINCT doc) FROM p WHERE lemma='migration'","kwic":"SELECT doc,sent,idx FROM p WHERE lemma='migration' LIMIT 50","pos_filter":"SELECT count(*) FROM p WHERE pos='NOUN'","dependency":"SELECT count(*) FROM p WHERE rel='nsubj'","per_document":"SELECT doc,count(*) FROM p GROUP BY doc"}
    return {k:timed(lambda q=q:con.execute(q).fetchall())[1] for k,q in queries.items()}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--tokens",type=int,default=1_000_000); p.add_argument("--out",type=Path,default=Path("results")); p.add_argument("--architectures",nargs="+",default=["sqlite_rows","sqlite_blobs","parquet_duckdb"]); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 result={"tokens":a.tokens,"architectures":{}}
 for name,fn,path in [("sqlite_rows",sqlite_rows,a.out/"rows.sqlite"),("sqlite_blobs",sqlite_blobs,a.out/"blobs.sqlite")]:
   if name not in a.architectures: continue
   gc.collect(); t=time.perf_counter(); queries=fn(path,a.tokens); write=round((time.perf_counter()-t)*1000,2)
   result["architectures"][name]={"write_plus_queries_ms":write,"size_bytes":path.stat().st_size,"queries":queries}
 # run fresh datasets separately to gather queries without duplicate schemas
 if "parquet_duckdb" in a.architectures:
   result["architectures"]["parquet_duckdb"]={"queries":parquet_duckdb(a.out/"tokens.parquet",a.tokens),"size_bytes":(a.out/"tokens.parquet").stat().st_size if (a.out/"tokens.parquet").exists() else None}
 if "sqlite_blobs_vectorized" in a.architectures:
   db, metrics = write_sqlite_blobs_vectorized(a.out/"blobs_vectorized.sqlite", a.tokens)
   print(json.dumps({"phase":"write_complete","architecture":"sqlite_blobs_vectorized",**metrics},indent=2), flush=True)
   query_started = time.perf_counter()
   query_result = sqlite_blob_frequency(db)
   query_ms = round((time.perf_counter() - query_started) * 1000, 2)
   db.close()
   result["architectures"]["sqlite_blobs_vectorized"]={**metrics,"queries":{"lemma_frequency":{"single_scan_ms":query_ms,**query_result}},"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 print(json.dumps(result,indent=2))
if __name__ == "__main__": main()
