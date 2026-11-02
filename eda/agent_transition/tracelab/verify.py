"""Independent, full-origin target/history/count checks against raw-call metadata."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'
d=pd.read_parquet(OUT/'calls_metadata.parquet')
d=d[d.provider.eq('codex')].sort_values('usage_report',kind='stable')
f=pd.read_parquet(OUT/'pilot_forecast_table.parquet')
config=json.loads((OUT/'pilot_config.json').read_text())
ts=d.usage_report.to_numpy(); origins=f.origin_utc.astype('int64').to_numpy()//10**9
result={}
for col,raw in [('input','input_tokens_total'),('output','output_tokens')]:
    cs=np.r_[0,d[raw].cumsum()]
    l,r=np.searchsorted(ts,origins,side='left'),np.searchsorted(ts,origins+600,side='left')
    expected=cs[r]-cs[l]
    assert np.array_equal(expected,f['target_'+col].to_numpy())
    result['full_target_'+col+'_checked']=len(origins)
    for w in [5,15,30,60]:
        l,r=np.searchsorted(ts,origins-w*60,side='left'),np.searchsorted(ts,origins,side='left')
        assert np.array_equal(cs[r]-cs[l],f[f'{col}_sum_{w}'].to_numpy())
for w in [5,15,30]:
    delta=np.zeros(len(f)+1,dtype=np.int64)
    for sid,g in d.groupby('session_id'):
        minutes=(g.usage_report.to_numpy()//60).astype('int64')
        spans=[]
        for a,b in zip(minutes+1,minutes+w):
            if spans and a<=spans[-1][1]+1: spans[-1][1]=max(spans[-1][1],b)
            else: spans.append([a,b])
        for a,b in spans:
            l=max(0,int(a-origins[0]//60)); r=min(len(f),int(b-origins[0]//60)+1)
            if l<r: delta[l]+=1;delta[r]-=1
    assert np.array_equal(delta.cumsum()[:-1],f[f'recent_reporting_sessions_{w}'].to_numpy())
    result[f'full_recent_session_{w}min_checked']=len(origins)
for idx in np.linspace(0,len(f)-1,9).astype(int):
    row=f.iloc[idx]; t=int(origins[idx]); past=d[d.usage_report<t]
    grouped=past.groupby('session_id')
    last=grouped.tail(1).set_index('session_id')
    active=last[last.usage_report>=t-1800]
    first=grouped.usage_report.min().reindex(active.index)
    count=grouped.size().reindex(active.index)
    diffs=past.assign(delta=past.groupby('session_id').input_tokens_total.diff()).groupby('session_id').tail(1).set_index('session_id').delta.reindex(active.index)
    vals={'session_recency_seconds':t-active.usage_report,
      'session_observed_age_seconds':t-first,'session_observed_report_count':count,
      'session_last_reported_input':active.input_tokens_total,'session_reported_input_change':diffs}
    for name,s in vals.items():
        for q,p in [('median',.5),('p90',.9)]:
            expected=s.quantile(p) if s.notna().any() else np.nan
            assert np.isclose(expected,row[name+'_'+q],equal_nan=True)
    recent=past[past.usage_report>=t-900].input_tokens_total
    expected=[recent.mean(),recent.std(ddof=0),recent.median(),recent.quantile(.9)]
    for name,v in zip(['mean','std','median','p90'],expected):
        assert np.isclose(v,row[f'input_len_{name}_15'],equal_nan=True)
result['independent_C_and_input_distribution_snapshots_checked']=9
assert (f.max_report_used<origins).all()
result['strict_feature_availability_all_origins']=True
for part,key in [('train','train_cutoff_utc'),('validation','test_start_utc')]:
    cutoff=pd.Timestamp(config[key]); p=f[f.split.eq(part)]
    assert (p.label_available_at_utc<=cutoff).all()
    result[part+'_labels_ready']=True
assert len(set(config['features']['A'])&set(config['features']['B_extra']))==0
assert len(set(config['features']['A'])&set(config['features']['C_extra']))==0
pred=pd.read_parquet(OUT/'pilot_predictions.parquet'); metrics=pd.read_csv(OUT/'pilot_metrics.csv')
for (target,model),g in pred.groupby(['target','model']):
    mae=(g.actual-g.prediction).abs().mean()
    saved=metrics[(metrics.target==target)&(metrics.model==model)&(metrics.split=='test')&(metrics.slice=='all')].mae.item()
    assert np.isclose(mae,saved)
result['test_metric_groups_recomputed']=10
result['scope']='Verification of observed-report task only; not verification of server arrivals, collection coverage, or predictive generalization.'
result['status']='PASS'
(OUT/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
