"""Temporal/coverage audit of the full public release metadata, no imputed arrivals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
d=pd.read_parquet(OUT/'calls_metadata.parquet')
records=[]
def longest(a):
    x=np.r_[False,np.asarray(a),False].astype(int)
    return int(np.max(np.flatnonzero(np.diff(x)==-1)-np.flatnonzero(np.diff(x)==1),initial=0))
for provider,g in [('all',d),*list(d.groupby('provider'))]:
    for clock in ['first_output','usage_report']:
        v=g.loc[g[clock].notna()].copy()
        if v.empty: continue
        v['minute']=(v[clock]//60).astype('int64')
        lo,hi=int(v.minute.min())+1,int(v.minute.max())
        ix=pd.RangeIndex(lo,hi)
        m=v.groupby('minute').agg(calls=('round_id','size'),input_tokens=('input_tokens_total','sum'),output_tokens=('output_tokens','sum')).reindex(ix,fill_value=0)
        q={'provider':provider,'clock':clock,'rows':len(v),'complete_interior_minutes':len(m),
           'nonempty_minutes':int((m.calls>0).sum()),'empty_minutes_pct':100*float((m.calls==0).mean()),
           'longest_empty_minutes':longest(m.calls==0),'longest_nonempty_minutes':longest(m.calls>0),
           'min_utc':pd.to_datetime(v[clock].min(),unit='s',utc=True).isoformat(),
           'max_utc':pd.to_datetime(v[clock].max(),unit='s',utc=True).isoformat()}
        cs=np.r_[0,m.calls.cumsum()]
        for h in [1,5,10]:
            y=cs[h:]-cs[:-h]
            q[f'zero_{h}min_target_pct']=100*float((y==0).mean())
        roll=m.calls.rolling(1440).sum().dropna()
        if len(roll):
            end=int(roll.idxmax())+1; start=end-1440; dense=m.loc[start:end-1]
            q['max_call_24h']={'start_utc':pd.to_datetime(start*60,unit='s',utc=True).isoformat(),
             'end_utc':pd.to_datetime(end*60,unit='s',utc=True).isoformat(),'calls':int(dense.calls.sum()),
             'empty_minutes_pct':float((dense.calls==0).mean()*100)}
        records.append(q)
pd.DataFrame(records).drop(columns='max_call_24h').to_csv(OUT/'time_grid.csv',index=False)
months=[]
for provider,g in d.groupby('provider'):
    g=g.copy(); g['month']=pd.to_datetime(g.first_output.fillna(g.usage_report),unit='s',utc=True).dt.strftime('%Y-%m')
    for month,v in g.groupby('month'):
        months.append(dict(provider=provider,month=month,calls=len(v),users=v.user.nunique(),sessions=v.session_id.nunique(),
          models=v.model.nunique(),largest_user_call_share=float(v.user.value_counts(normalize=True).iloc[0]),
          largest_model_call_share=float(v.model.value_counts(normalize=True).iloc[0])))
pd.DataFrame(months).to_csv(OUT/'monthly_composition.csv',index=False)
valid=d[d.first_output.notna()]
cols=['provider','user','first_output','last_output','input_tokens_total','output_tokens']
sus=valid[valid.duplicated(cols,keep=False)]
delay=d.usage_report-d.last_output
gap=d.sort_values(['provider','user','session_id','first_output']).groupby(['provider','user','session_id']).first_output.diff().dropna()
z={'time_grids':records,'codex_report_delay_gt_60s':int((delay>60).sum()),'codex_report_delay_gt_600s':int((delay>600).sum()),
   'codex_report_delay_gt_day':int((delay>86400).sum()),'coincident_output_token_fingerprint_rows':len(sus),
   'coincident_output_token_fingerprint_extra_rows':int(valid.duplicated(cols).sum()),
   'coincident_output_across_distinct_sessions_groups':int((sus.groupby(cols).session_id.nunique()>1).sum()),
   'observed_followup_pairs':len(gap),'observed_followup_gap_seconds':{str(k):float(v) for k,v in gap.quantile([0,.5,.9,.99,1]).items()},
   'observed_followup_within_minutes_pct':{str(h):float((gap<=h*60).mean()*100) for h in [1,5,10]},
   'interpretation':'Sparse retrospective donation, unknown collection coverage. Contiguous calendar bins are not evidence of continuous logging. Fingerprints are suspicious coincidences, not proven duplicates.'}
repeated={}
for provider,g in d.groupby('provider'):
    k=['user','round_id']; u=g[g.duplicated(k,False)]; groups=u.groupby(k)
    repeated[provider]={'extra_rows':int(g.duplicated(k).sum()),'groups':groups.ngroups,
      'groups_across_sessions':int((groups.session_id.nunique()>1).sum()),
      'groups_varying_input_tokens':int((groups.input_tokens_total.nunique()>1).sum()),
      'groups_varying_output_tokens':int((groups.output_tokens.nunique()>1).sum()),
      'groups_varying_first_output':int((groups.first_output.nunique()>1).sum())}
    if provider=='codex':
        repeated[provider]['rows_by_report_month']=pd.to_datetime(u.usage_report,unit='s',utc=True).dt.strftime('%Y-%m').value_counts().sort_index().to_dict()
        start=pd.Timestamp('2026-05-01',tz='UTC').timestamp(); end=pd.Timestamp('2026-06-01',tz='UTC').timestamp()
        pilot=g[g.usage_report.ge(start)&g.usage_report.lt(end)]
        affected=set(u.session_id)&set(pilot.session_id)
        repeated[provider]['repeated_round_rows_reported_in_pilot_month']=int((u.usage_report.ge(start)&u.usage_report.lt(end)).sum())
        repeated[provider]['pilot_sessions_with_suspect_earlier_reports']=len(affected)
        repeated[provider]['pilot_rows_in_those_sessions']=int(pilot.session_id.isin(affected).sum())
z['cross_session_round_identity_repeats']=repeated
(OUT/'temporal.json').write_text(json.dumps(z,indent=2,ensure_ascii=False))
print(json.dumps(z,indent=2))
