"""Inspect one public Parquet shard; do not print or export conversation content."""
from pathlib import Path
import hashlib,json
from collections import Counter
import pyarrow.parquet as pq
import pandas as pd
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source/train-00000-of-00039.parquet'
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
digest=hashlib.sha256(SRC.read_bytes()).hexdigest()
assert digest=='5cfde1c01460d2f7d90fdc6c245278cee20f32101d5275b6cf3d06f30a5f8f60'
rows=pq.read_table(SRC).to_pylist()
spans=[]; attrkeys=Counter(); spankeys=Counter(); benches=Counter(); harnesses=Counter()
for r in rows:
    benches[r['benchmark']]+=1; harnesses[r['harness']]+=1
    for s in r['spans']:
        a=s.get('attributes') or {}; attrkeys.update(a.keys()); spankeys.update(s.keys())
        spans.append(dict(session_id=r['session_id'],span_id=s.get('span_id'),trace_id=s.get('trace_id'),
          parent_span_id=s.get('parent_span_id'),start_time=s.get('start_time'),end_time=s.get('end_time'),
          input_tokens=a.get('gen_ai.usage.input_tokens'),output_tokens=a.get('gen_ai.usage.output_tokens'),
          operation=a.get('gen_ai.operation.name'),model=a.get('gen_ai.request.model'),collected_at=r['collected_at']))
d=pd.DataFrame(spans)
raw_missing_start=int(d.start_time.isna().sum())
raw_missing_end=int(d.end_time.isna().sum())
d['start_time']=pd.to_datetime(d.start_time,utc=True,errors='coerce',format='mixed')
d['end_time']=pd.to_datetime(d.end_time,utc=True,errors='coerce',format='mixed')
d.to_parquet(OUT/'sample_spans_metadata.parquet',index=False)
llm=d[d.input_tokens.notna()|d.output_tokens.notna()]
j={'scope':'ONE shard (00 of 39), not full dataset; no SWE-bench rows in this shard.',
 'revision':'6b1add7c19f1fb50bb0edf5b240d6149a5c621fb','url':'https://huggingface.co/datasets/DiscoPosse/agent-llm-traces/resolve/6b1add7c19f1fb50bb0edf5b240d6149a5c621fb/data/train-00000-of-00039.parquet',
 'bytes':SRC.stat().st_size,'sha256':digest,'full_distribution_shards':39,'full_distribution_bytes':983592848,
 'rows_sessions':len(rows),'spans':len(d),'llm_token_spans':len(llm),'benchmarks':dict(benches),'harnesses':dict(harnesses),
 'span_keys':dict(spankeys),'attribute_keys':dict(attrkeys),'unique_span_ids':int(d.span_id.nunique()),
 'parent_span_id_nonnull':int(d.parent_span_id.notna().sum()),
 'start_utc':d.start_time.min().isoformat(),'end_utc':d.end_time.max().isoformat(),
 'missing_start':raw_missing_start,'missing_end':raw_missing_end,
 'invalid_start':int(d.start_time.isna().sum())-raw_missing_start,'invalid_end':int(d.end_time.isna().sum())-raw_missing_end,
 'end_before_start':int((d.end_time<d.start_time).sum()),
 'input_missing_llm':int(llm.input_tokens.isna().sum()),'output_missing_llm':int(llm.output_tokens.isna().sum()),
 'input_zero_llm':int((llm.input_tokens==0).sum()),'output_zero_llm':int((llm.output_tokens==0).sum()),
 'negative_input_llm':int((llm.input_tokens<0).sum()),'negative_output_llm':int((llm.output_tokens<0).sum()),
 'input_tokens_llm_sum':int(llm.input_tokens.sum()),'output_tokens_llm_sum':int(llm.output_tokens.sum()),
 'semantic_caveats':['Span starts are client-operation times, not verified server arrival.',
 'Token attributes are final span attributes; do not expose input_tokens before end_time without additional instrumentation.',
 'Benchmark schedules are not real user demand. collected_at is export/collection time, not request time.',
 'This sampled shard has no parent_span_id field, contrary to README example.']}
(OUT/'audit.json').write_text(json.dumps(j,indent=2))
print(json.dumps({k:v for k,v in j.items() if k not in ['attribute_keys','span_keys']},indent=2))
