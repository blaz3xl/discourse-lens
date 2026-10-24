"""Vectorized Parquet/Zstd writer; intentionally separate from row-wise SQLite."""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VOCAB=np.array(["migration","policy","community","border","worker","public","social","evidence","debate","agency"])
POS=np.array(["NOUN","VERB","ADJ","ADP","DET"]); REL=np.array(["nsubj","obj","amod","obl","root"])
SURFACE_LENGTHS=np.array([len(value) for value in VOCAB],dtype=np.int64)
SURFACE_PREFIX=np.concatenate((np.array([0],dtype=np.int64),np.cumsum(SURFACE_LENGTHS[:-1]+1)))
CYCLE_WIDTH=int(np.sum(SURFACE_LENGTHS+1))
def batch(start,n):
 i=np.arange(start,start+n,dtype=np.int64); vocab_idx=i%len(VOCAB); surface=VOCAB[vocab_idx]
 source_start=(i//len(VOCAB))*CYCLE_WIDTH+SURFACE_PREFIX[vocab_idx]
 return pa.table({"doc":i//1000,"sent":i//20,"idx":i%20,"surface":surface,"lemma":surface,"pos":POS[i%5],"morph":np.full(n,"Number=Sing"),"head":np.maximum(0,i%20-1),"rel":REL[i%5],"start":source_start,"end":source_start+SURFACE_LENGTHS[vocab_idx]})
def ms(fn):
 t=time.perf_counter(); x=fn(); return x,round((time.perf_counter()-t)*1000,1)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--tokens",type=int,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--batch",type=int,default=100_000); a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
 writer=None; t=time.perf_counter()
 for start in range(0,a.tokens,a.batch):
  table=batch(start,min(a.batch,a.tokens-start))
  if writer is None: writer=pq.ParquetWriter(a.out,table.schema,compression="zstd")
  writer.write_table(table)
 writer.close(); write_ms=round((time.perf_counter()-t)*1000,1)
 con=duckdb.connect(config={"threads":"2","memory_limit":"512MB"}); con.from_parquet(str(a.out)).create_view("p")
 q={"lemma_frequency":"SELECT count(*) FROM p WHERE lemma='migration'","document_frequency":"SELECT count(DISTINCT doc) FROM p WHERE lemma='migration'","pos_filter":"SELECT count(*) FROM p WHERE pos='NOUN'","kwic_candidates":"SELECT doc,sent,idx FROM p WHERE lemma='migration' LIMIT 50","per_document":"SELECT doc,count(*) FROM p GROUP BY doc","dependency":"SELECT count(*) FROM p WHERE rel='nsubj'","lemma_pos":"SELECT count(*) FROM p WHERE lemma='migration' AND pos='NOUN'","document_lemma":"SELECT count(*) FROM p WHERE doc=1 AND lemma='migration'"}
 out={}
 for name,sql in q.items():
  _cold,cold_ms=ms(lambda s=sql:con.execute(s).fetchall()); warms=[ms(lambda s=sql:con.execute(s).fetchall())[1] for _ in range(3)]; out[name]={"cold_ms":cold_ms,"warm_median_ms":sorted(warms)[1]}
 row_count=con.execute("SELECT count(*) FROM p").fetchone()[0]
 print(json.dumps({"tokens":a.tokens,"write_ms":write_ms,"size_bytes":a.out.stat().st_size,"validation":{"row_count":row_count,"row_count_matches":row_count==a.tokens},"queries":out,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"python":platform.python_version(),"duckdb":duckdb.__version__,"pyarrow":pa.__version__,"batch_size":a.batch,"duckdb_threads":2,"duckdb_memory_limit":"512MB"},indent=2))
if __name__=="__main__": main()
