"""Audit only public TraceLab metadata; never print message/tool content.
Run: /opt/anaconda3/bin/python3 audit.py (paths relative to this file).
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import gzip, hashlib, json, platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'source/syfi_coding_trace.jsonl.gz'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
EXPECTED = '9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b'
digest = hashlib.sha256(SRC.read_bytes()).hexdigest()
assert digest == EXPECTED
keys, event_types, event_sources, zones = Counter(), Counter(), Counter(), Counter()
event_keys, tool_keys = Counter(), Counter()
token_fields = ['input_tokens_total', 'output_tokens', 'prefix_tokens', 'newly_append_tokens',
 'reasoning_output_tokens', 'claude_uncached_input_tokens',
 'claude_cache_creation_input_tokens', 'claude_cache_read_input_tokens']
raw_seen, identity_seen, trace_seen = set(), set(), set()
issues = Counter()
rows = []
tool_stats = Counter()
def stamp(v):
    if v is None: return np.nan
    try:
        d = datetime.fromisoformat(v.replace('Z', '+00:00'))
        if d.tzinfo is None:
            issues['naive_timestamp'] += 1
            return np.nan
        zones[str(d.utcoffset())] += 1
        return d.timestamp()
    except (ValueError, TypeError):
        issues['invalid_timestamp'] += 1
        return np.nan
def lo(a): return min(a) if a else np.nan
def hi(a): return max(a) if a else np.nan
with gzip.open(SRC, 'rb') as f:
    for line in f:
        r = json.loads(line)
        keys.update(r.keys())
        rh = hashlib.sha256(line.rstrip()).digest()
        ident = (r.get('provider'), r.get('session_id'), r.get('round_id'))
        issues['duplicate_complete_rows'] += rh in raw_seen
        issues['duplicate_provider_session_round'] += ident in identity_seen
        issues['duplicate_trace_key'] += r.get('trace_key') in trace_seen
        raw_seen.add(rh); identity_seen.add(ident); trace_seen.add(r.get('trace_key'))
        events = []
        for e in r.get('timing_events', []):
            event_keys.update(e.keys())
            typ = e.get('event_type'); event_types[(r['provider'], typ)] += 1
            event_sources[(r['provider'], typ, e.get('source'))] += 1
            t = stamp(e.get('timestamp'))
            if np.isfinite(t): events.append((typ, t))
            else: issues['event_without_valid_timestamp'] += 1
        allts = [t for _,t in events]
        outts = [t for k,t in events if k in ('text', 'reasoning', 'tool_call')]
        ints = [t for k,t in events if k in ('user_message', 'tool_result')]
        reports = [t for k,t in events if k == 'usage_report']
        firstout = lo(outts)
        before = [t for t in ints if t <= firstout] if outts else []
        issues['event_list_not_chronological'] += any(b<a for a,b in zip(allts,allts[1:]))
        tools = r.get('tools') or []
        emitted, results = [], []
        for tool in tools:
            tool_keys.update(tool.keys())
            a, b = stamp(tool.get('emitted_at')), stamp(tool.get('result_at'))
            if np.isfinite(a): emitted.append(a)
            if np.isfinite(b): results.append(b)
            tool_stats['records'] += 1
            tool_stats['missing_emitted'] += not np.isfinite(a)
            tool_stats['missing_result'] += not np.isfinite(b)
            tool_stats['negative_latency'] += np.isfinite(a) and np.isfinite(b) and b<a
            tool_stats['error_true'] += tool.get('is_error') is True
            tool_stats['error_unknown'] += tool.get('is_error') is None
        z = {k:r.get(k) for k in ['provider','user','session_id','round_id','round_index','turn_id','model','store']+token_fields}
        z.update(first_event=lo(allts), last_event=hi(allts), first_output=firstout,
          last_output=hi(outts), input_ready_proxy=hi(before), last_input_event=hi(ints),
          usage_report=hi(reports), n_events=len(events), n_user=sum(k=='user_message' for k,t in events),
          n_tool_result=sum(k=='tool_result' for k,t in events), n_tools=len(tools),
          first_tool=lo(emitted), last_tool_result=hi(results),
          input_events_after_first_output=sum(t>firstout for t in ints) if outts else 0)
        rows.append(z)
df = pd.DataFrame(rows)
df.to_parquet(OUT / 'calls_metadata.parquet', index=False)
summary = dict(source={'release':'v0.0.1','url':'https://github.com/uw-syfi/TraceLab/releases/download/v0.0.1/syfi_coding_trace.jsonl.gz',
 'bytes':SRC.stat().st_size,'sha256':digest,'digest_matches_github':True},
 environment={'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__},
 rows=len(df), fields=dict(keys), event_keys=dict(event_keys), tool_keys=dict(tool_keys),
 timestamp_offsets=dict(zones), issues=dict(issues), tools=dict(tool_stats), providers={})
def stats(a):
    s=pd.Series(a).dropna()
    if s.empty: return {'n':0}
    return {'n':len(s), 'min':float(s.min()),'median':float(s.median()),
      'p90':float(s.quantile(.9)),'p99':float(s.quantile(.99)),'max':float(s.max())}
def iso(t): return pd.to_datetime(t, unit='s', utc=True).isoformat() if pd.notna(t) else None
for p,g in df.groupby('provider'):
    q={'rows':len(g),'users':g.user.nunique(),'sessions':g.session_id.nunique(),'models':g.model.nunique(),
      'time_min_utc':iso(g.first_event.min()),'time_max_utc':iso(g.last_event.max()),'tokens':{}}
    for c in token_fields:
        s=g[c]; q['tokens'][c]={'missing':int(s.isna().sum()),'zero':int((s==0).sum()),
          'negative':int((s<0).sum()),'noninteger':int(((s.dropna()%1)!=0).sum()),
          'sum':int(s.sum()), **stats(s)}
    q['split_identity_mismatch']=int((g.input_tokens_total != g.prefix_tokens+g.newly_append_tokens).sum())
    q['reasoning_gt_output']=int((g.reasoning_output_tokens>g.output_tokens).sum())
    q['both_tokens_zero']=int(((g.input_tokens_total==0)&(g.output_tokens==0)).sum())
    q['timing_missing']={c:int(g[c].isna().sum()) for c in ['first_event','first_output','last_output','input_ready_proxy','usage_report','last_tool_result']}
    q['input_events_after_first_output_rows']=int((g.input_events_after_first_output>0).sum())
    q['input_ready_to_first_output_seconds']=stats(g.first_output-g.input_ready_proxy)
    q['output_to_usage_report_seconds']=stats(g.usage_report-g.last_output)
    q['usage_report_after_last_tool_result']=int((g.usage_report>=g.last_tool_result).sum())
    q['input_decreases_from_previous_round']=int((g.sort_values(['user','session_id','round_index']).groupby(['user','session_id']).input_tokens_total.diff()<0).sum())
    q['turn_id_nonnull']=int(g.turn_id.notna().sum())
    if p=='claude':
        q['claude_input_components_mismatch']=int((g.input_tokens_total != g.claude_uncached_input_tokens+g.claude_cache_creation_input_tokens+g.claude_cache_read_input_tokens).sum())
    summary['providers'][p]=q
pd.DataFrame([dict(provider=p,event_type=k,count=n) for (p,k),n in event_types.items()]).to_csv(OUT/'event_types.csv',index=False)
pd.DataFrame([dict(provider=p,event_type=k,source=s,count=n) for (p,k,s),n in event_sources.items()]).to_csv(OUT/'event_sources.csv',index=False)
df['date_utc']=pd.to_datetime(df.first_output.fillna(df.last_event),unit='s',utc=True).dt.strftime('%Y-%m-%d')
df.groupby(['date_utc','provider']).agg(calls=('session_id','size'),input_tokens=('input_tokens_total','sum'),
 output_tokens=('output_tokens','sum'),users=('user','nunique'),sessions=('session_id','nunique'),models=('model','nunique')).reset_index().to_csv(OUT/'daily.csv',index=False)
df.groupby(['provider','model'],dropna=False).agg(calls=('session_id','size'),input_tokens=('input_tokens_total','sum'),output_tokens=('output_tokens','sum'),users=('user','nunique')).reset_index().to_csv(OUT/'models.csv',index=False)
df.groupby(['provider','user']).agg(calls=('session_id','size'),first=('first_event','min'),last=('last_event','max'),sessions=('session_id','nunique')).reset_index().to_csv(OUT/'participant_spans.csv',index=False)
summary['session_user_conflicts']=int((df.groupby(['provider','session_id']).user.nunique()>1).sum())
summary['date_rows']=int(df.date_utc.nunique())
summary['continuous_collection']='Not established: no collector heartbeat/coverage schedule in release rows; empty bins are absence of recorded events only.'
summary['availability']='Codex: explicit usage_report; Claude: no explicit final usage timestamp, last_output is only an optimistic proxy. Neither input_ready_proxy nor first_output is server arrival.'
(OUT/'audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item()))
print(json.dumps({k:summary[k] for k in ['rows','issues','tools']},default=lambda x:x.item()))
for p,q in summary['providers'].items(): print(p, {k:v for k,v in q.items() if k!='tokens'})
